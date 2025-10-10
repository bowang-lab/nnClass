import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetDatasetBlosc2CLS
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, nnUNetDataLoaderCLS, nnUNetDataBalancedLoaderCLS
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from sklearn import metrics
import wandb
from monai.networks.nets import DenseNet121, SEResNet50, ViT, SwinUNETR, MedNeXt


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * torch.pow((1 - pt), self.gamma)
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

    
class DynamicClassificationLoss(nn.Module):
    def __init__(self, class_num=2, imbalanced=False):
        super(DynamicClassificationLoss, self).__init__()
        self.class_num = class_num
        self.imbalanced = imbalanced
        if class_num == 2:
            if imbalanced:
                self.loss = FocalBCEWithLogitsLoss()
            else:
                self.loss = nn.BCEWithLogitsLoss()
        else:
            if imbalanced:
                self.loss = FocalCEWithLogitsLoss()
            else:
                self.loss = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        """
        Computes the classification loss.
        :param logits: model outputs
        :param targets: ground truth labels
        :return: classification loss
        """
        if self.class_num == 2:
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if targets.type() != logits.type():
                targets = targets.type(logits.type())
        return self.loss(logits, targets)
class DynamicLossWeightUpdater:
    def __init__(self, window_size=5, momentum=0.9):
        self.window_size = window_size
        self.seg_loss_history = []
        self.momentum = momentum
        self.seg_weight = 1.0
        self.cls_weight = 0.0

    def _compute_slope(self):
        """Computes the slope (rate of change) of seg_loss over the recent window."""
        if len(self.seg_loss_history) < 2:
            return 0.0
        x = list(range(len(self.seg_loss_history)))
        y = self.seg_loss_history
        # simple linear regression slope
        n = len(x)
        avg_x = sum(x) / n
        avg_y = sum(y) / n
        num = sum((x[i] - avg_x) * (y[i] - avg_y) for i in range(n))
        den = sum((x[i] - avg_x)**2 for i in range(n))
        slope = num / den if den != 0 else 0
        return slope

    def update(self, current_seg_loss):
        """Updates weights based on seg_loss trend."""
        self.seg_loss_history.append(current_seg_loss)
        if len(self.seg_loss_history) > self.window_size:
            self.seg_loss_history.pop(0)

        slope = self._compute_slope()

        # Example logic:
        # if the slope is small (flat curve), we can shift focus to classification
        if abs(slope) < 0.01:
            self.cls_weight = min(0.9, self.momentum * self.cls_weight + (1 - self.momentum) * 0.9)
        else:
            self.cls_weight = self.momentum * self.cls_weight  # decay if seg is still learning

        self.seg_weight = 1.0 - self.cls_weight  # keep sum = 1

        return self.seg_weight, self.cls_weight, slope

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, fusion_layers=3):
        super().__init__()
        
        # Reduce channels for each level
        if fusion_layers == 3:
            self.conv1x1_1 = nn.Conv3d(in_channels_list[0], out_channels, kernel_size=1)
            self.deconv1 = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)
            )
            self.smooth1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_2 = nn.Conv3d(in_channels_list[1], out_channels, kernel_size=1)
        self.conv1x1_3 = nn.Conv3d(in_channels_list[2], out_channels, kernel_size=1)
        
        # Transposed convolutions for upsampling
        self.deconv2 = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2)
        )

        

        
        # Additional convolutions after addition
        self.smooth2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Normalization and activation
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fusion_layers = fusion_layers

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize FPN specific weights with custom strategies
        """
        # Initialize 1x1 convolutions
        conv_list = [self.conv1x1_1, self.conv1x1_2, self.conv1x1_3] if self.fusion_layers == 3 else [self.conv1x1_2, self.conv1x1_3]
        deconv_list = [self.deconv1, self.deconv2] if self.fusion_layers == 3 else [self.deconv2]
        smooth_list = [self.smooth1, self.smooth2] if self.fusion_layers == 3 else [self.smooth2]
        for m in conv_list:
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize deconvolution layers
        for m in deconv_list:
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize smoothing convolutions
        for m in smooth_list:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3):
        # Forward pass implementation remains the same
        p3 = self.conv1x1_3(x3)
        p2 = self.conv1x1_2(x2)
        p3_up = self.deconv2(p3)
        p2 = self.relu(self.norm(p2 + p3_up))
        p2 = self.smooth2(p2)
        if self.fusion_layers == 3:
            p1 = self.conv1x1_1(x1)
            p2_up = self.deconv1(p2)
            p1 = self.relu(self.norm(p1 + p2_up))
            p1 = self.smooth1(p1)
            return p1
        else:
            return p2
        

class SegmentationNetworkFusionClassificationHead(nn.Module):
    def __init__(self, seg_network: nn.Module, features_per_stage: List[int],
                 num_hidden_features: int, num_classes: int, fusion_layers: int = 3):
        super().__init__()
        self.seg_network = seg_network
        assert hasattr(self.seg_network, 'encoder')
        self.encoder = self.seg_network.encoder

        self.feature_fusion_block = FeaturePyramidNetwork(features_per_stage[-3:], num_hidden_features, fusion_layers)
        # Post FPN processing
        self.conv_block = nn.Sequential(
            nn.Conv3d(num_hidden_features, num_hidden_features*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_hidden_features*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_hidden_features*2, num_hidden_features*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_hidden_features*2),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden_features*2, num_hidden_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_hidden_features, num_classes)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        #self.freeze_classification_head()
        #self.freeze_seg_network()



    def forward(self, x):
        skips = self.seg_network.encoder(x)
        x = self.feature_fusion_block(skips[-3], skips[-2], skips[-1])
        x = self.conv_block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ViT3D(nn.Module):
    """
    3D Vision Transformer for medical image classification using MONAI
    """
    def __init__(self, img_size=(96, 96, 96), patch_size=(16, 16, 16), in_channels=1, 
                 num_classes=2, hidden_size=768, mlp_dim=3072, num_heads=12, 
                 num_layers=12, dropout_rate=0.1):
        super().__init__()
        
        self.network = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=3,
            classification=True,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.network(x)[0]  # Get classification output



class SwinClassificationNetwork(nn.Module):
    def __init__(self, in_channels, num_classes, depths, num_heads, 
                embed_dim, dropout_rate):
        super().__init__()
        
        # Swin backbone
        self.swin_backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,  # This will be overridden
            depths=depths,
            num_heads=num_heads,
            feature_size=48,
            norm_name="instance",
            drop_rate=dropout_rate,
            attn_drop_rate=dropout_rate,
            dropout_path_rate=0.1,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3
        )
        
        # Classification head
        feature_size = embed_dim * 2**(len(depths)-1)  # 96 * 8 = 768 for default config
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, num_classes)
        )
        
    def forward(self, x):
        # Extract features using Swin encoder
        hidden_states_out = self.swin_backbone.swinViT(x, normalize=True)
        
        # Use the last encoder output for classification
        enc4 = hidden_states_out[4]  # Shape: [B, H*W*D/64, 768]
        #print(enc4.shape)
        # Reshape to spatial dimensions for pooling
        # B, L, C = enc4.shape
        # # Calculate spatial dimensions
        # H = W = D = int(np.ceil(L**(1/3)))  # Approximate cube root
        
        # # Reshape and pool
        # enc4 = enc4.transpose(1, 2).view(B, C, H, W, D)
        pooled = self.global_avg_pool(enc4)  # [B, C, 1, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Classification
        output = self.classifier(pooled)
        return output

class MedNeXtClassifier(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=1, num_classes=1, dropout=0.2):
        super().__init__()
        # Use MedNeXt as feature extractor
        self.backbone = MedNeXt(
            spatial_dims=spatial_dims,
            init_filters=16,
            in_channels=in_channels,
            out_channels=32,  # Use fewer channels for feature extraction
        )
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Extract features using MedNeXt backbone
        features = self.backbone(x)
        
        # Global average pooling: [B, C, H, W, D] -> [B, C, 1, 1, 1]
        pooled = self.global_pool(features)
        
        # Flatten: [B, C, 1, 1, 1] -> [B, C]
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classification: [B, C] -> [B, num_classes]
        output = self.classifier(flattened)
        
        return output

class nnUNetREGTrainer(nnUNetTrainer):
    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()
            self._best_mae = None
            self.enable_deep_supervision = False
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.cls_class_num = 1
            cls_head_output = self.cls_class_num if self.cls_class_num > 2 else 1
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
                self.configuration_manager.network_arch_init_kwargs['features_per_stage'][-1],
                cls_head_output,
                self.configuration_manager.patch_size
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.dataset_name = self.preprocessed_dataset_folder.split('/')[-2]
            wandb.init(
                project=f"REG_{self.dataset_name}",
                name=f"{self.__class__.__name__}_fold{self.fold}",
            )

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def get_cls_class_num(self, df_path: str) -> int:
        """
        Returns the number of classes for classification
        """
        df = pd.read_csv(df_path)

        return df['label'].nunique()
    
    def _build_regression_loss(self):

        loss = nn.MSELoss()
        return loss

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1,
                                   patch_size=None) -> nn.Module:

        segmentation_network = nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)

        return SegmentationNetworkFusionClassificationHead(segmentation_network,
                                                         arch_init_kwargs["features_per_stage"],
                                                         emb_dim, cls_class_num)

    def set_deep_supervision_enabled(self, enabled: bool):
        if enabled:
            raise NotImplementedError("Deep supervision is not implemented.")
        else:
            pass
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()
        self.val_cases = len(val_keys)
        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDatasetBlosc2CLS(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = nnUNetDatasetBlosc2CLS(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        self.regression_loss = self._build_regression_loss()

        dl_tr = nnUNetDataLoaderCLS(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoaderCLS(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        self.num_val_iterations_per_epoch = self.val_cases // self.batch_size
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        cls_label = batch['cls_label']

        data = data.to(self.device, non_blocking=True)
        cls_label = cls_label.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            cls_output = self.network(data)
            # del data
            seg_loss = 0
            reg_loss = self.regression_loss(cls_output, cls_label.unsqueeze(1).float())
        l = reg_loss


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy(),
                'seg_loss': seg_loss,
                'reg_loss': reg_loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            cls_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            dist.all_gather_object(seg_losses_tr, outputs['seg_loss'])
            dist.all_gather_object(cls_losses_tr, outputs['reg_loss'])
            loss_here = np.vstack(losses_tr).mean()
            seg_loss_here = np.vstack(seg_losses_tr).mean()
            cls_loss_here = np.vstack(cls_losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            seg_loss_here = np.mean(outputs['seg_loss'])
            cls_loss_here = np.mean(outputs['reg_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        #self.seg_weight, self.cls_weight, slope = self.dynamic_loss_weight_updater.update(seg_loss_here)
        wandb.log({
            'train_regression_loss': cls_loss_here,
            # 'seg_loss_slope': slope,
            # 'seg_weight': self.seg_weight,
            # 'cls_weight': self.cls_weight
        })


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        cls_label = batch['cls_label']
        ids = batch['keys']
        all_probs = []
        all_preds = []
        all_labels = []
        all_ids = []
        all_ids.extend(ids)

        data = data.to(self.device, non_blocking=True)
        cls_label = cls_label.to(self.device, non_blocking=True)


        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            cls_output = self.network(data)
            del data
            seg_loss = 0
            reg_loss = self.regression_loss(cls_output, cls_label.unsqueeze(1).float())
        l = reg_loss
        
        
        

        all_preds.extend(cls_output.cpu().numpy())  # Move to CPU and convert to numpy
        all_labels.extend(cls_label.cpu().numpy())
        


        return {'total_loss': l.detach().cpu().numpy(),
                'all_labels': all_labels,
                'all_preds': all_preds,
                'all_ids': all_ids
                }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)


        if self.is_ddp:
            world_size = dist.get_world_size()

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['total_loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['total_loss'])

        all_preds = torch.tensor(np.array(outputs_collated['all_preds']))
        all_labels = torch.tensor(np.array(outputs_collated['all_labels']))
        all_ids = outputs_collated['all_ids']

        mae = nn.L1Loss()(all_preds, all_labels.float()).item()
        mean_mae = np.mean(mae)

        wandb.log({
            'val/MSE': loss_here,
            "val/MAE": mean_mae,
        })

        self.epoch_df = pd.DataFrame({
            'ids': all_ids,
            'preds': all_preds.cpu().numpy().tolist(),
            'labels': all_labels.cpu().numpy().tolist(),
        })

        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_mae', mean_mae, self.current_epoch)


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('val_mae', np.round(self.logger.my_fantastic_logging['val_mae'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['val_losses'][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['val_losses'][-1]
            self.print_to_log_file(f"Yayy! New best EMA MSE: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_bestmse.pth'))
        
        if self._best_mae is None or self.logger.my_fantastic_logging['val_mae'][-1] < self._best_mae:
            self._best_mae = self.logger.my_fantastic_logging['val_mae'][-1]
            self.print_to_log_file(f"Yayy! New best mae: {np.round(self._best_mae, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_bestmae.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

class DenseNetREGTrainer(nnUNetREGTrainer):

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1) -> nn.Module:

        

        return DenseNet121(
            spatial_dims=3,
            in_channels=num_input_channels,
            out_channels=cls_class_num,
            dropout_prob=0.2
        )

class SEResNetREGTrainer(nnUNetREGTrainer):

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1,
                                   patch_size=None) -> nn.Module:

        return SEResNet50(
            spatial_dims=3,
            in_channels=num_input_channels,
            num_classes=cls_class_num
        )

class ViTREGTrainer(nnUNetREGTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1,
                                   patch_size=None) -> nn.Module:

        return ViT3D(
            img_size=patch_size,
            patch_size=(16, 16, 16),
            in_channels=num_input_channels,
            num_classes=cls_class_num,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            dropout_rate=0.1
        )

class SwinViTREGTrainer(nnUNetREGTrainer):

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1,
                                   patch_size=None) -> nn.Module:

        return SwinClassificationNetwork(
            in_channels=num_input_channels,
            num_classes=cls_class_num,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96,
            dropout_rate=0.1
        )

class MedNeXtREGTrainer(nnUNetREGTrainer):

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   emb_dim: int = 256,
                                   cls_class_num: int = 1,
                                   patch_size=None) -> nn.Module:

        return MedNeXtClassifier(
            spatial_dims=3,
            in_channels=num_input_channels,
            num_classes=cls_class_num,
        )
