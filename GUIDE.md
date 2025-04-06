# Audio Deepfake Detection: Implementation Guide

This guide provides detailed instructions for implementing and using the AASIST audio deepfake detection system, optimized for systems with limited hardware resources.

## Quick Start

If you've already downloaded the required dataset files, you can run the entire pipeline with one command:

```
.\run.bat
```

This will automatically:
1. Prepare the dataset
2. Preprocess the data
3. Train and evaluate the model

## Project Overview

The project implements a modified version of the AASIST (Attention-based Audio Spoofing and Injection Spectrogram Transformer) model for detecting AI-generated speech. The system has been optimized to work on limited hardware resources.

## System Requirements

- Python 3.7+
- CPU only is fine (GPU is optional and can speed up training)
- At least 4GB of RAM (8GB recommended)
- Approximately 2GB of free disk space for the minimal setup

## Detailed Step-by-Step Guide

### Step 1: Set Up the Environment

1. Create and activate a virtual environment:
   ```
   # With Anaconda
   conda create -n deepfake_audio python=3.8
   conda activate deepfake_audio
   
   # With venv (Windows)
   python -m venv deepfake_audio_env
   deepfake_audio_env\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Step 2: Download and Prepare the Dataset

1. Create the necessary directories:
   ```
   mkdir -p data/raw data/processed models/checkpoints results
   ```

2. Download the minimum required files from ASVspoof5:
   - Go to https://zenodo.org/records/14498691
   - Download:
     - `ASVspoof5_protocols.tar.gz` (20.7 MB)
     - `flac_D_ab.tar` (6.6 GB) - This is one part of the development set
   - Place these files in the `data/raw` directory

3. Extract and prepare the dataset:
   ```
   python utils/prepare_dataset.py
   ```

### Step 3: Preprocess the Dataset

Run the optimized preprocessing script:
```
python utils/preprocess_data.py --subset dev --limit 500 --low_memory
```

This will:
- Process only the development set
- Limit to 500 samples for faster processing
- Use low memory mode to avoid crashes on systems with limited RAM

### Step 4: Train the Model

Train a smaller, optimized model:
```
python notebooks/aasist_training.py --small_model --epochs 5 --batch_size 16 --max_samples 500 --device cpu --test
```

Options explained:
- `--small_model`: Use a smaller architecture without graph attention
- `--epochs 5`: Train for only 5 epochs
- `--batch_size 16`: Use a small batch size to save memory
- `--max_samples 500`: Limit the number of samples
- `--device cpu`: Force CPU usage even if a GPU is available
- `--test`: Evaluate the model after training

### Step 5: Run Inference on Audio Files

Test the model on audio files:
```
python utils/simplified_inference.py --input_file path/to/audio_file.wav --force_cpu
```

## Troubleshooting Common Issues

### Dataset Preparation Issues

- **"No dataset files found"**: Make sure the dataset files are in the `data/raw` directory with the correct names
- **Error during extraction**: Use smaller dataset files like `flac_D_ab.tar` instead of the full dataset
- **Protocol file not found**: Make sure you've downloaded `ASVspoof5_protocols.tar.gz`

### Preprocessing Issues

- **Memory errors**: Use the `--low_memory` flag and limit the number of samples with `--limit`
- **Processing errors**: Try using a different subset with `--subset dev` or `--subset eval`
- **No audio files found**: Check the dataset extraction path and structure

### Training Issues

- **Out of memory errors**: Reduce batch size further with `--batch_size 8` or `--batch_size 4`
- **Slow training**: Reduce the number of epochs and samples, use `--small_model`
- **Model doesn't learn**: Increase learning rate with `--learning_rate 0.001`

## Interview Preparation Guide

Here are key topics to understand for your interview:

### 1. Technical Understanding of AASIST Model

- **AASIST Architecture**: Explain how the model combines spectral and temporal features using attention mechanisms
- **Graph Attention Networks**: How they capture relationships between different frequency bands
- **Raw Waveform vs. Spectrograms**: Understand the trade-offs between using raw audio and pre-extracted features
- **Optimizations Made**: Be able to explain what changes were made to make the model work on limited hardware

### 2. Audio Deepfake Detection Fundamentals

- **Common Artifacts in Synthetic Speech**: Understand what patterns distinguish fake from real audio
- **Feature Extraction**: How audio is converted to spectrograms or processed as raw waveforms
- **Frequency Domain Analysis**: How spectral features help identify synthetic speech

### 3. Training Process and Evaluation Metrics

- **Loss Functions**: Why cross-entropy loss is suitable for this binary classification task
- **Evaluation Metrics**: Understand EER (Equal Error Rate), ROC curves, and why they're important
- **Dataset Balancing**: Why balanced datasets are important and how undersampling helps

### 4. Implementation and Optimization Decisions

- **Hardware Constraints**: Be able to discuss how you addressed limited hardware resources
- **Memory Usage Optimization**: How batching, worker count, and model size affect memory usage
- **Training Time vs. Accuracy**: Discuss the trade-offs between speed and performance

### 5. Real-World Applications and Limitations

- **Generalization to Unseen Attacks**: How models might perform on new, previously unseen deepfake methods
- **Real-Time Processing Challenges**: Discuss the challenges of implementing real-time detection
- **Deployment Considerations**: Model size, computation requirements, and practical concerns

### Sample Interview Questions

1. "Walk me through the AASIST model architecture and how it detects audio deepfakes."
2. "What modifications did you make to the original model to optimize it for limited hardware?"
3. "What are the most common artifacts found in synthetic speech, and how does your model detect them?"
4. "How would your approach perform on different types of synthetic speech not present in the training data?"
5. "What evaluation metrics did you use, and why are they appropriate for this task?"
6. "If you had more time and resources, what improvements would you make to the model?"

## Next Steps for Improvement

If you want to enhance the system further:

1. **Incremental Training**: Train on more data by processing and training incrementally
2. **Data Augmentation**: Implement pitch shifting, time stretching, and noise addition
3. **Model Compression**: Apply techniques like quantization and pruning
4. **Transfer Learning**: Pre-train on larger speech datasets
5. **Feature Engineering**: Experiment with different audio feature extraction methods

By following this guide and understanding these key concepts, you'll be well-prepared for your interview on audio deepfake detection. 