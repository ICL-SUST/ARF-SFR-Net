# ARF-SFR-Net

This repository contains the implementation of FARF-SFR-Net for fine-grained few-shot image classification.

## Installation

### Environment Setup

Create a conda environment with the required dependencies:

```bash
conda env create -f environment.yaml
conda activate ARF-SFR
```

### Requirements

- Python 3.7+
- PyTorch 1.7.0+
- CUDA 11.0+ (for GPU acceleration)
- Additional dependencies: torchvision, numpy, scipy, pillow, tensorboard

## Dataset Preparation

### Supported Datasets
1. **CUB-200-2011**: Fine-grained bird species classification
   - [Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)

2. **Stanford Cars**: Vehicle classification
   - [Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)

3. **Stanford Dogs**: Dog breed classification
   - [Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)

4. **Meta-iNat & Tiered-Meta-iNat**: Natural species classification variants

### Data Organization

After downloading, organize datasets in the following structure:
```
./datasets/
├── CUB_fewshot_cropped/
├── cars/
├── dogs/
├── meta-iNat/
└── tiered-meta-iNat/
```

The train/validation/test splits follow the specifications in `split.txt`.

## Training

### CUB-200-2011 with Conv-4 Backbone

```bash
cd experiments/CUB_fewshot_cropped/ARF-SFR/Conv-4
./train.sh
```

### CUB-200-2011 with ResNet-12 Backbone

```bash
cd experiments/CUB_fewshot_cropped/ARF-SFR/ResNet-12
./train.sh
```

### Other Datasets

Similar training scripts are available for other datasets:

```bash
# Stanford Cars
cd experiments/cars/ARF-SFR/[Conv-4|ResNet-12]
./train.sh

# Stanford Dogs
cd experiments/dogs/ARF-SFR/[Conv-4|ResNet-12]
./train.sh

# Meta-iNaturalist
cd experiments/meta-iNat/ARF-SFR/Conv-4
./train.sh
```

## Evaluation

### Testing Trained Models

```bash
# Conv-4 backbone
cd experiments/CUB_fewshot_cropped/ARF-SFR/Conv-4
python test.py

# ResNet-12 backbone
cd experiments/CUB_fewshot_cropped/ARF-SFR/ResNet-12
python test.py
```

## Model Components

### Core Modules

- `models/ARF.py`: Main ARF Strategy implementation
- `models/ARFSFR.py`: Main ARF-SFR-Net implementation
- `models/ARFSFR_Conv_4.py`: Conv-4 backbone with ARF-SFR components
- `models/ARFSFR_ResNet.py`: ResNet-12 backbone with ARF-SFR components

### Training Infrastructure

- `trainers/arfsfr_train.py`: ARF-SFR-Net specific training logic
- `trainers/trainer.py`: General training framework
- `datasets/`: Data loading and preprocessing utilities