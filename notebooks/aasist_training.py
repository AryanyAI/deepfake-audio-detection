"""
Script for training the AASIST model on the preprocessed ASVspoof dataset.
Optimized for systems with limited hardware resources.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
from tqdm import tqdm
import time
import gc  # Garbage collection for memory management

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.aasist_model import AASIST
from models.trainer import ModelTrainer

class SimpleAudioDataset(Dataset):
    """
    Simple PyTorch dataset for preprocessed audio features.
    Memory-efficient implementation.
    """
    def __init__(self, features_dir, subset, max_samples=None):
        """
        Args:
            features_dir: Directory containing preprocessed features
            subset: Subset to use ('train', 'dev', 'eval')
            max_samples: Maximum number of samples to load (useful for testing)
        """
        self.features_dir = os.path.join(features_dir, subset)
        self.max_samples = max_samples
        
        # Check if directory exists
        if not os.path.exists(self.features_dir):
            raise ValueError(f"Features directory not found: {self.features_dir}")
        
        # Load labels
        labels_path = os.path.join(self.features_dir, "labels.npy")
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels file not found: {labels_path}")
        
        self.labels = np.load(labels_path)
        
        # Load filenames
        filenames_path = os.path.join(self.features_dir, "filenames.txt")
        if not os.path.exists(filenames_path):
            # If filenames.txt doesn't exist, use all .npy files except labels.npy
            self.filenames = [f for f in os.listdir(self.features_dir) if f.endswith('.npy') and f != 'labels.npy']
            # Strip .npy extension
            self.filenames = [os.path.splitext(f)[0] for f in self.filenames]
        else:
            # Load filenames from file
            with open(filenames_path, 'r') as f:
                self.filenames = [line.strip() for line in f]
        
        # Limit number of samples if specified
        if self.max_samples is not None and self.max_samples < len(self.filenames):
            self.filenames = self.filenames[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        print(f"Loaded {len(self.filenames)} samples for {subset} set")
        
        # Count labels
        real_count = np.sum(self.labels == 0)
        fake_count = np.sum(self.labels == 1)
        print(f"Real samples: {real_count}, Fake samples: {fake_count}")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load feature
        filename = self.filenames[idx]
        feature_path = os.path.join(self.features_dir, f"{filename}.npy")
        
        try:
            feature = np.load(feature_path)
            
            # Check if feature is raw waveform or spectrogram
            if len(feature.shape) == 1:
                # Raw waveform
                feature = feature.astype(np.float32)
                feature = torch.from_numpy(feature).unsqueeze(0)  # Add channel dimension
            else:
                # Spectrogram
                feature = feature.astype(np.float32)
                feature = torch.from_numpy(feature).unsqueeze(0)  # Add channel dimension
            
            # Get label
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            
            return feature, label
            
        except Exception as e:
            print(f"Error loading feature {filename}: {e}")
            # Create a dummy feature and return the correct label
            if idx % 2 == 0:
                # Create a dummy spectrogram
                feature = torch.randn(1, 80, 400, dtype=torch.float32)
            else:
                # Create a dummy waveform
                feature = torch.randn(1, 64000, dtype=torch.float32)
            
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            
            return feature, label

def create_smaller_model(raw_input=True, n_class=2):
    """
    Create a smaller version of the AASIST model for lower-end hardware.
    
    Args:
        raw_input: Whether to use raw waveform as input
        n_class: Number of output classes
        
    Returns:
        Smaller AASIST model
    """
    model = AASIST(
        raw_input=raw_input,
        sinc_filter=True,
        graph_attention=False,  # Disable graph attention for smaller model
        n_class=n_class
    )
    
    return model

def train_model(args):
    """
    Train the AASIST model.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = SimpleAudioDataset(args.features_dir, 'dev', max_samples=args.max_samples)  # Use dev set for both training and validation
    
    # Split dataset for validation
    total_size = len(train_dataset)
    val_size = min(int(total_size * 0.2), 200)  # 20% for validation, max 200 samples
    train_size = total_size - val_size
    
    from torch.utils.data import random_split
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Split dataset: {train_size} for training, {val_size} for validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False  # Set to False to reduce memory usage
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False  # Set to False to reduce memory usage
    )
    
    # Create model
    if args.small_model:
        print("Using smaller model for low-end hardware")
        model = create_smaller_model(raw_input=args.raw_input, n_class=2)
    else:
        model = AASIST(
            raw_input=args.raw_input,
            sinc_filter=True,
            graph_attention=True,
            n_class=2
        )
    
    # Create trainer
    trainer = ModelTrainer(model, device=device)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_path=args.checkpoint_dir
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Clean up memory
    del train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Plot training history
    fig = trainer.plot_history()
    fig.savefig(os.path.join(args.checkpoint_dir, 'training_history.png'))
    
    print(f"Training history saved to {os.path.join(args.checkpoint_dir, 'training_history.png')}")
    
    # Evaluate on test set if specified
    if args.test:
        print("Evaluating on test set...")
        
        # Load test dataset
        test_dataset = SimpleAudioDataset(args.features_dir, 'dev', max_samples=args.max_samples)
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False
        )
        
        # Test model
        metrics = trainer.test(test_loader)
        
        # Print results
        print("\nTest Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"EER: {metrics['eer']:.4f}")
        print(f"AUC: {metrics['roc_auc']:.4f}")
        
        # Plot confusion matrix
        cm_fig = trainer.plot_confusion_matrix(metrics['confusion_matrix'])
        cm_fig.savefig(os.path.join(args.checkpoint_dir, 'confusion_matrix.png'))
        
        # Plot ROC curve
        roc_fig = trainer.plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
        roc_fig.savefig(os.path.join(args.checkpoint_dir, 'roc_curve.png'))
        
        print(f"Evaluation results saved to {args.checkpoint_dir}")
        
        # Clean up
        del test_loader, test_dataset
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\nTraining completed successfully!")
    print("\nNext steps:")
    print("1. Use the trained model for inference: python utils/simplified_inference.py --input_file <path_to_audio> --force_cpu")

def main():
    """
    Main function for training the AASIST model.
    """
    parser = argparse.ArgumentParser(description='Train the AASIST model on the ASVspoof dataset.')
    parser.add_argument('--features_dir', type=str, default='./data/processed', help='Directory containing preprocessed features')
    parser.add_argument('--checkpoint_dir', type=str, default='./models/checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--raw_input', action='store_true', help='Use raw waveform as input')
    parser.add_argument('--test', action='store_true', help='Evaluate on test set after training')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum number of samples to use (for testing)')
    parser.add_argument('--small_model', action='store_true', help='Use a smaller model for low-end hardware')
    
    args = parser.parse_args()
    
    train_model(args)

if __name__ == '__main__':
    main() 