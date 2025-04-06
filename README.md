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
git clone https://github.com/yourusername/audio-deepfake-detection.git
cd audio-deepfake-detection

# Install dependencies
pip install -r requirements.txt

# Download required dataset files (manual step)
# Download from https://zenodo.org/records/14498691
# - ASVspoof5_protocols.tar.gz
# - flac_D_ab.tar
# Place these files in the data/raw directory

# Run the entire pipeline
.\run.bat
```

## Dataset

This project uses the ASVspoof5 dataset. Due to the size of the full dataset, the implementation is optimized to work with a subset:

- Development set (`flac_D_ab.tar`) - One part of the development set (6.6 GB)
- Protocol files (`ASVspoof5_protocols.tar.gz`) - Contains labels and evaluation protocols (20.7 MB)

These files can be downloaded from [Zenodo](https://zenodo.org/records/14498691).

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

- Original ASVspoof5 dataset: [Zenodo](https://zenodo.org/records/14498691)
- Wang, Xin, et al. "ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech." arXiv preprint arXiv:2502.08857 (2024). 