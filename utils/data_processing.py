"""
Utility functions for processing audio data for deepfake detection.
"""

import os
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

def load_audio(file_path, sr=16000):
    """
    Load an audio file and resample it to the specified sampling rate.
    
    Args:
        file_path (str): Path to the audio file
        sr (int): Target sampling rate
        
    Returns:
        numpy.ndarray: Audio signal
    """
    try:
        # Load audio file
        signal, fs = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(signal.shape) > 1:
            signal = signal[:, 0]
        
        # Resample if needed
        if fs != sr:
            signal = librosa.resample(signal, orig_sr=fs, target_sr=sr)
            
        return signal
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

def extract_mel_spectrogram(audio, sr=16000, n_mels=80):
    """
    Extract Mel spectrogram features from audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sampling rate
        n_mels (int): Number of Mel bands
        
    Returns:
        numpy.ndarray: Mel spectrogram
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=1024, 
        hop_length=256,
        win_length=1024
    )
    # Convert to log scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

def pad_or_truncate(audio, target_length=64000):
    """
    Pad or truncate audio to a fixed length.
    
    Args:
        audio (numpy.ndarray): Audio signal
        target_length (int): Target length in samples
        
    Returns:
        numpy.ndarray: Padded or truncated audio
    """
    if len(audio) > target_length:
        # Truncate
        return audio[:target_length]
    else:
        # Pad with zeros
        padded = np.zeros(target_length)
        padded[:len(audio)] = audio
        return padded

def create_dataset(audio_dir, labels_file, output_dir, sample_rate=16000, duration=4.0):
    """
    Process audio files and save features for training.
    
    Args:
        audio_dir (str): Directory containing audio files
        labels_file (str): Path to the file mapping audio files to labels
        output_dir (str): Directory to save processed features
        sample_rate (int): Target sampling rate
        duration (float): Target duration in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read labels
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename, label = parts[0], int(parts[1])
                labels[filename] = label
    
    # Calculate target length
    target_length = int(duration * sample_rate)
    
    features = []
    file_labels = []
    
    # Process each audio file
    for filename in tqdm(os.listdir(audio_dir)):
        if filename not in labels:
            continue
            
        file_path = os.path.join(audio_dir, filename)
        label = labels[filename]
        
        # Load and preprocess audio
        audio = load_audio(file_path, sr=sample_rate)
        if audio is None:
            continue
            
        # Pad or truncate to fixed length
        audio = pad_or_truncate(audio, target_length)
        
        # Extract features
        mel_spec = extract_mel_spectrogram(audio, sr=sample_rate)
        
        # Save features
        np.save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy"), mel_spec)
        
        features.append(mel_spec)
        file_labels.append(label)
    
    # Save labels
    np.save(os.path.join(output_dir, "labels.npy"), np.array(file_labels))
    
    return features, file_labels

class AudioDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for audio deepfake detection.
    """
    def __init__(self, features_dir, transform=None):
        """
        Args:
            features_dir (str): Directory containing preprocessed features
            transform (callable, optional): Optional transform to apply to features
        """
        self.features_dir = features_dir
        self.transform = transform
        
        # Load labels
        self.labels = np.load(os.path.join(features_dir, "labels.npy"))
        
        # Get file list
        self.file_list = [f for f in os.listdir(features_dir) if f.endswith('.npy') and f != 'labels.npy']
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.features_dir, self.file_list[idx])
        
        # Load feature
        feature = np.load(file_path)
        
        # Apply transform if specified
        if self.transform:
            feature = self.transform(feature)
        
        # Convert to torch tensor
        feature = torch.from_numpy(feature).float()
        
        # Get label
        label = self.labels[idx]
        
        return feature, label 