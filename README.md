# nnUNetCLS

nnUNetCLS is a Python-based project that extends the nnUNet framework with joint segmentation and classification capabilities. It combines the power of nnUNet's segmentation architecture with multi-class classification, making it ideal for medical imaging tasks that require both pixel-level segmentation and image-level classification predictions.

## Features

- **Joint Architecture**: Simultaneous segmentation and classification in a single model
- **Stratified Data Splitting**: Advanced data splitting with demographic stratification (age, gender)
- **Multi-Modal Input**: Support for multi-channel medical images (CT, PET, MRI)
- **Flexible Classification**: Support for both binary and multi-class classification tasks
- **Comprehensive Inference**: Batch processing with sliding window prediction and test-time augmentation

## Installation

To set up nnUNetCLS, first install the required dependencies:

```bash
pip install wandb
pip install -e .
```

## Configure nnUNet Paths
Before using nnUNetCLS, you need to configure the nnUNet environment paths. Modify the paths in nnunetv2/paths.py to point to your desired directories:
```python
# Edit nnunetv2/paths.py
nnUNet_raw = "/path/to/your/nnUNet_raw"
nnUNet_preprocessed = "/path/to/your/nnUNet_preprocessed" 
nnUNet_results = "/path/to/your/nnUNet_results"
```
## Usage

### 1. Prepare Data

## Preprocessing with nnUNet

nnUNetCLS relies on **nnUNet’s preprocessing pipeline** to standardize image spacing, intensity normalization, and patch extraction. Preprocessing must be completed before training or inference.  

### 1. Required Data Structure  

Your dataset must follow the **nnUNet folder convention**:  
```bash
nnUNet_raw/
└── Dataset<DATASET_ID>_<DATASET_NAME>/
├── imagesTr/ # Training images (NIfTI format)
│ ├── PatientID_0000.nii.gz # First modality (e.g., CT)
│ ├── PatientID_0001.nii.gz # Second modality (e.g., MRI)
│ └── ...
├── labelsTr/ # Training labels (segmentation masks)
│ ├── PatientID.nii.gz
│ └── ...
├── imagesTs/ # Test images (no labels required)
│ ├── TestID_0000.nii.gz
│ └── ...
└── dataset.json # Dataset description file
```

**Notes:**  
- Each modality is indexed as `_0000`, `_0001`, etc.  
- Segmentation labels must have the same base name as the training images (without modality suffix).  
- `dataset.json` defines modalities, labels, and dataset splits.  

---

### 2. Default Preprocessing  

Run the standard nnUNet preprocessing:  

```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID> -c 3d_fullres --verify_dataset_integrity

```

For Res Encoder
```bash
nnUNetv2_plan_experiment -d <DATASET_ID> -pl nnUNetPlannerResEncM #nnUNetPlannerResEncL / nnUNetPlannerResEncXL
```

Use `generate_cls_data.py` to create stratified train/validation/test splits from your clinical dataset:

```bash
python generate_cls_data.py \
    --input_path /path/to/clinical_data.csv \
    --output_path /path/to/output/folder \
    --identifier_column PatientID \
    --label_column diagnosis
```

**Arguments:**
- `--input_path, -i`: Path to CSV/Excel file containing clinical and imaging information
- `--output_path, -o`: Directory to save classification data and splits -> "/path/to/your/nnUNet_preprocessed" 
- `--identifier_column, -id`: Column name for patient identifiers (default: 'patient_id')
- `--label_column, -label`: Column name for classification labels (default: 'label')

**Required CSV columns:**
- Patient identifiers (e.g., 'PatientID')
- Classification labels 
- `Age_at_StudyDate`: For age-based stratification
- `Gender`: For gender-based stratification

**Outputs:**
- `cls_data.csv`: Classification dataset
- `test_data.csv`: Held-out test set (20% of data)
- `splits_final.json`: 5-fold cross-validation splits with stratification
- Automatic filtering of cases without segmentation data

### 2. Training

```bash
nnUNetv2_train 161 3d_fullres 0 -tr <TrainerName>
nnUNetv2_train 714 3d_fullres all -p nnUNetResEncUNetMPlans
```
**Notes:**
for baseline cls support
- MedNeXtTrainer
- ViTTrainer
- DenseNetTrainer
- SEResNetTrainer
- SwinViTTrainer
- nnUNetREGTrainer
- MedNeXtREGTrainer
- ViTREGTrainer
- DenseNetREGTrainer
- SEResNetREGTrainer
- SwinViTREGTrainer
define in nnunetv2/training/nnUNetTrainer/nnUNetCLSTrainer.py

### 3. Inference

Run joint segmentation and classification inference on NIfTI images:

```bash
python nnunet_cls_infer_nii.py \
    --input_path /path/to/input/images/ \
    --output_path /path/to/output/ \
    --model_path /path/to/trained/model \
    --fold all \
    --checkpoint checkpoint_best.pth \
    --device cuda \
    --cls_mode mean
```

**Arguments:**
- `--input_path, -i`: Directory containing input NIfTI images (expects `*_000X.nii.gz` naming convention)
- `--output_path, -o`: Directory to save segmentation masks and classification results
- `--model_path`: Path to trained nnUNet model directory
- `--fold`: Fold number or 'all' for ensemble prediction (default: 'all')
- `--checkpoint`: Checkpoint filename (default: 'checkpoint_best.pth')
- `--use_softmax`: Apply softmax to segmentation output (default: False)
- `--device`: Computing device ('cuda' or 'cpu', default: 'cuda')
- `--cls_mode`: Classification aggregation mode ('mean' or 'weighted', default: 'mean')

**Input Format:**
Images should follow nnUNet naming convention:
- `PatientID_0000.nii.gz` (first modality)
- `PatientID_0001.nii.gz` (second modality)
- etc.

**Outputs:**
- `{PatientID}.nii.gz`: Segmentation masks for each case
- `results.csv`: Classification probabilities for all cases

### 4. Key Features

**Stratified Cross-Validation:**
- Creates balanced splits based on age quartiles, gender, and target labels
- Ensures representative distribution across all folds
- 80/20 train-test split with 5-fold cross-validation on training data

**Advanced Inference:**
- Sliding window prediction with Gaussian weighting
- Test-time augmentation with mirroring
- Multi-fold model ensembling
- Memory-efficient processing for large images
- Automatic batch processing of multiple cases

**Classification Modes:**
- `mean`: Average classification scores across all patches
- `weighted`: Weight classification by segmentation confidence

## Model Architecture

The framework extends nnUNet with:
- Shared encoder for both segmentation and classification
- Dual output heads (segmentation + classification)
- Feature aggregation from the final encoder stage
- Support for both binary and multi-class classification

## License

This project is licensed under the [Apache License 2.0](https://github.com/ChingYuanYu/nnunetcls/blob/main/LICENSE).

## Citation

If you use nnUNetCLS in your research, please cite the original nnUNet paper and this extension.

---

**Note:** Ensure your input data follows the nnUNet preprocessing requirements and naming conventions for optimal performance.