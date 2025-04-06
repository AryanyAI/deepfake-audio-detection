# Audio Deepfake Detection

This project implements an optimized version of the AASIST (Attention-based Audio Spoofing and Injection Spectrogram Transformer) model for detecting AI-generated human speech. The system is specifically designed to work efficiently on limited hardware resources.

## Overview

The detection system uses a combination of convolutional neural networks and attention mechanisms to identify subtle artifacts present in synthetic speech. The model has been optimized to run on systems with limited resources while maintaining good detection performance.

## Features

- **Hardware Efficient**: Optimized to run on CPU with minimal RAM requirements
- **Model Adaptations**: Modified AASIST architecture for lower computational demands
- **Simple Pipeline**: Streamlined data preparation, preprocessing, and training workflow
- **Robust Detection**: Effectively identifies common audio deepfake patterns

## Quick Start

```
# Clone the repository
git clone https://github.com/AryanyAI/deepfake-audio-detection.git
cd audio-deepfake-detection

# Install dependencies
pip install -r requirements.txt

# Download required dataset files (manual step)
# Download from https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
# - ASVspoof5_protocols.tar.gz
# - ASVspoof2019_LA_dev.zip
# Place these files in the data/raw directory

# Run the entire pipeline
.\run.bat
```

## Dataset

This project uses the ASVspoof 2019 Logical Access (LA) dataset. The implementation is optimized to work with a subset, specifically the development set (`LA_dev`).

- **Source Used:** ASVspoof 2019 Dataset on Kaggle: [https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)
- **Files Required for this Project:**
    - `ASVspoof2019_LA_dev.zip` (or extracted equivalent containing `.flac` files)
    - `ASVspoof2019_LA_cm_protocols.zip` (or extracted equivalent containing `.trl.txt` label files)

These files should be downloaded from the Kaggle link above and placed/extracted into the `data/raw` directory according to the structure expected by `utils/prepare_dataset.py` and `utils/preprocess_data.py` (typically placing the protocol `.txt` files in `data/raw/protocols/` and the audio `.flac` files in a path like `data/raw/LA_dev/.../flac/`).

*Note: Other sources like Zenodo also host versions of this dataset, but the Kaggle source was used for this specific implementation run.*

## System Requirements

- Python 3.7+
- CPU (GPU optional)
- 4GB+ RAM (8GB recommended)
- ~2GB disk space for minimal setup

## Implementation Details

The implementation follows these main steps:
1. **Dataset Preparation**: Extract and organize the ASVspoof5 dataset
2. **Preprocessing**: Convert audio files to feature representations
3. **Model Training**: Train an optimized version of the AASIST model
4. **Evaluation**: Test performance on held-out test data
5. **Inference**: Analyze new audio files for authenticity

## Documentation

- **GUIDE.md**: Detailed explanation of the implementation and interview preparation
- **run.bat**: Windows batch script to run the entire pipeline
- **utils/**: Utility scripts for dataset preparation and processing
- **models/**: Model architecture and training code
- **notebooks/**: Training scripts

## References
- Original ASVspoof 2019 dataset: [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336)
- Original ASVspoof5 dataset: [Zenodo](https://zenodo.org/records/14498691)
- Wang, Xin, et al. "ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech." arXiv preprint arXiv:2502.08857 (2024). 