import numpy as np
import torch
from torch._dynamo import OptimizedModule
import itertools
from time import time
import os
import SimpleITK as sitk
import nnunetv2
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from typing import Tuple, Union
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.resampling.default_resampling import fast_resample_logit_to_shape
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from tqdm import tqdm
import argparse
import glob
import os
import gc
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import json
from tqdm import tqdm
import re
from collections import defaultdict

def logit_to_segment(predicted_logits):
    max_logit, max_class = torch.max(predicted_logits, dim=0)
                
                # Apply threshold: Only assign the class if its logit exceeds the threshold
    segmentation = torch.where(max_logit >= 0.5, max_class, torch.tensor(0, device=predicted_logits.device))

    return segmentation

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                use_softmax,
                                                                return_probabilities: bool = False,
                                                                ):

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]



    # apply_inference_nonlin will convert to torch
    if properties_dict['shape_after_cropping_and_before_resampling'][0] < 600:
        predicted_logits = fast_resample_logit_to_shape(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
        gc.collect()
        empty_cache(predicted_logits.device)
        if use_softmax:
            predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)

            del predicted_logits
            
            # Start timing for converting probabilities to segmentation
            segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
        else:
            # Get the class with the maximum logit at each pixel
            segmentation = logit_to_segment(predicted_logits)

    else:
        print(f"Predicted Logits: {predicted_logits.shape}")
        segmentation = fast_resample_logit_to_shape(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])



    dtype = torch.uint8 if len(label_manager.foreground_labels) < 255 else torch.uint16
    segmentation_reverted_cropping = torch.zeros(properties_dict['shape_before_cropping'], dtype=dtype)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation

    del segmentation

    # Revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.permute(plans_manager.transpose_backward)

    return segmentation_reverted_cropping.cpu()

class SimplePredictor(nnUNetPredictor):
    """
    simple predictor for nnUNet
    """
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str,
                                             cls_mode: str = 'mean'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None
            ckpt = checkpoint['network_weights']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            num_classes = ckpt['classifier.3.weight'].shape[0]
            parameters.append(ckpt)

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')

        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer.'
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,
        emb_dim=configuration_manager.network_arch_init_kwargs["features_per_stage"][-1],
        cls_class_num=num_classes
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.num_classes = num_classes
        print(f'Using {self.num_classes} classes for classification')
        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        # network.load_state_dict(parameters[0])
        # for params in self.list_of_parameters:
        #     self.network.load_state_dict(params)
        
        
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.cls_mode = cls_mode
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def preprocess(self, image, props):
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        #image = torch.from_numpy(image).to(dtype=torch.float32, memory_format=torch.contiguous_format).to(self.device)
        data = preprocessor.run_case_npy(image,
                                                  None,
                                                  props,
                                                  self.plans_manager,
                                                  self.configuration_manager,
                                                  self.dataset_json)
        data = torch.from_numpy(data[0]).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        return data
    
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction, cls_prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction_flipped, cls_prediction_flipped = self.network(torch.flip(x, axes))
                prediction += torch.flip(prediction_flipped, axes)
                cls_prediction += cls_prediction_flipped
            prediction /= (len(axes_combinations) + 1)
            cls_prediction /= (len(axes_combinations) + 1)
        return prediction, cls_prediction
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        class_logits = None
        results_device = self.device if do_on_device else torch.device('cpu')
        self.network = self.network.to(self.device)
        self.network.eval()
        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            # TODO: The shape should be number of classes
            class_logits = torch.zeros((self.num_classes), dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            cls_slices = 0
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction, class_logits_patch = self._internal_maybe_mirror_and_predict(workon)
                prediction = prediction[0].to(results_device)
                class_logits_patch = class_logits_patch[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

                if self.cls_mode == 'mean':
                    class_logits += class_logits_patch
                    cls_slices += 1
                else:
                    if prediction.max() > 0.5:
                        class_logits += class_logits_patch
                        cls_slices += 1
            print(f'Number of slices used for classification: {cls_slices}')
            predicted_logits /= n_predictions
            class_logits /= cls_slices
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            del class_logits
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits, class_logits
    

    def inference(self, image, properties_dict, use_softmax):
        image = self.preprocess(image, properties_dict)
        predicted_logits = None

        with torch.no_grad():
            assert isinstance(image, torch.Tensor)
            empty_cache(self.device)

            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

                data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
                for params in self.list_of_parameters:
                    # messing with state dict names...
                    if not isinstance(self.network, OptimizedModule):
                        self.network.load_state_dict(params)
                    else:
                        self.network._orig_mod.load_state_dict(params)
                    if predicted_logits is None:
                        predicted_logits, cls_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                    self.perform_everything_on_device)
                        if self.num_classes == 1:
                            predicted_logits, cls_logits = predicted_logits.to('cpu'), torch.sigmoid(cls_logits).cpu()
                            print(torch.sigmoid(cls_logits).cpu())
                        else:    
                            predicted_logits, cls_logits = predicted_logits.to('cpu'), cls_logits.to('cpu')
                        
                    else:
                        cur_prediction, cur_class_logit = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                    self.perform_everything_on_device)
                        predicted_logits += cur_prediction.to('cpu')
                        if self.num_classes == 1:
                            cls_logits += torch.sigmoid(cur_class_logit).cpu()
                            print(torch.sigmoid(cur_class_logit).cpu())
                        else:
                            cls_logits += cur_class_logit.to('cpu')

                if len(self.list_of_parameters) > 1:
                    predicted_logits /= len(self.list_of_parameters)
                    cls_logits /= len(self.list_of_parameters)
                empty_cache(self.device) # Start time for inference time calculation
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

                segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,
                                                                self.plans_manager,
                                                                self.configuration_manager,
                                                                self.label_manager,
                                                                properties_dict,
                                                                use_softmax,
                                                                return_probabilities=False,
                                                                )


        if self.num_classes > 1:
            cls_probs = torch.softmax(cls_logits, dim=0).cpu().numpy()
        else:
            cls_probs = cls_logits

        return segmentation, cls_probs

def group_images_by_idx(input_folder):
    pattern = re.compile(r"^(.*)_000\d+\.nii\.gz$")  # Captures everything before the _000x
    grouped = defaultdict(list)

    for entry in os.scandir(input_folder):
        if entry.name.endswith(".nii.gz"):
            match = pattern.match(entry.name)
            if match:
                idx = match.group(1)
                grouped[idx].append(entry.path)

    # Optional: sort each group to ensure consistent order
    for paths in grouped.values():
        paths.sort()

    return dict(grouped)

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Inference for nnUNet model")
        parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image file')
        parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output segmentation')
        parser.add_argument('--model_path', type=str, required=True, help='Name of the model to use for inference')
        parser.add_argument('--fold', type=str, default=(0,1,2,3,4), help='Fold number to use for inference (default: 0)')
        parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth', help='Path to the model checkpoint file')
        parser.add_argument('--use_softmax', default=False, help='Apply softmax to the output probabilities')
        parser.add_argument("--device", type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
        parser.add_argument('--cls_mode', type=str, default='mean', choices=['mean', 'weighted'], help='Classification mode: mean or weighted')

        return parser.parse_args()

    args = parse_arguments()

    if args.device == 'cpu':
        perform_everything_on_device = False
    else:
        perform_everything_on_device = True

    device = torch.device(args.device, 0)
    predictor = SimplePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=perform_everything_on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        args.model_path,
        use_folds= args.fold,
        checkpoint_name= args.checkpoint,
        cls_mode=args.cls_mode
    )
    predictor.network.to(device)

    input_folder = args.input_path
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    test_ids = []
    test_probs = []
    cases_dict = group_images_by_idx(input_folder)
    for case in tqdm(cases_dict.keys(), desc="Processing cases"):
        segmentation_path = os.path.join(output_folder, f"{case}.nii.gz")
        print(f"Processing case: {case}")
        case_path = cases_dict[case]
        test_ids.append(case)
        image, props = SimpleITKIO().read_images(case_path)
        segmentation, cls_probs = predictor.inference(image, props, args.use_softmax)
        sitk_img = sitk.GetImageFromArray(segmentation)
        sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
        sitk_img.SetOrigin(props['sitk_stuff']['origin'])
        sitk_img.SetDirection(props['sitk_stuff']['direction'])
        #sitk.WriteImage(sitk_img, segmentation_path)
        test_probs.append(cls_probs.tolist())
    
    results = pd.DataFrame({'identifier': test_ids, 'probs': test_probs})
    results.to_csv(os.path.join(output_folder, 'results.csv'), index=False)





