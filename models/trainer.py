"""
Trainer module for the AASIST model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns

class ModelTrainer:
    """
    Trainer class for the AASIST model.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The AASIST model to train
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Initialize training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_eer': None
        }
    
    def train(self, train_loader, val_loader=None, epochs=10, learning_rate=0.001, weight_decay=1e-5, save_path=None):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
            save_path: Path to save the model
        """
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, labels in pbar:
                # Move data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})
            
            # Calculate epoch statistics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if save_path is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': best_val_loss,
                    }, os.path.join(save_path, 'best_model.pth'))
                
                # Print epoch statistics
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            else:
                # Print epoch statistics
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
        
        # Save final model
        if save_path is not None:
            torch.save({
                'epoch': epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': train_loss,
            }, os.path.join(save_path, 'final_model.pth'))
            
        return self.history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            loss, accuracy
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Move data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate statistics
        val_loss = val_loss / len(data_loader.dataset)
        val_acc = val_correct / val_total
        
        return val_loss, val_acc
    
    def test(self, test_loader):
        """
        Test the model on a dataset and compute metrics.
        
        Args:
            test_loader: DataLoader for the test dataset
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                # Move data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                scores = torch.softmax(outputs, dim=1)[:, 1]  # Probability of being fake
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)
        
        # Check if we have both classes in the test set
        has_both_classes = len(np.unique(all_labels)) > 1
        
        # Initialize metrics
        metrics = {}
        
        # Compute basic metrics
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        
        # Handle case where there's only one class
        if has_both_classes:
            # Compute more advanced metrics when both classes are present
            metrics['precision'] = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            metrics['f1'] = f1_score(all_labels, all_preds, average='binary', zero_division=0)
            
            # Compute EER (Equal Error Rate)
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            fnr = 1 - tpr
            
            # Handle potential NaN values
            valid_indices = ~np.isnan(np.absolute(fnr - fpr))
            if np.any(valid_indices):
                eer_idx = np.nanargmin(np.absolute(fnr - fpr))
                eer_threshold = thresholds[eer_idx]
                eer = fpr[eer_idx]
            else:
                # If no valid indices, use placeholder values
                eer_threshold = 0.5
                eer = 0.5
            
            # Compute AUC (Area Under the ROC Curve)
            roc_auc = auc(fpr, tpr)
            
            # Store EER in history
            self.history['test_eer'] = eer
            
            # Add to metrics
            metrics['eer'] = eer
            metrics['eer_threshold'] = eer_threshold
            metrics['roc_auc'] = roc_auc
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        else:
            # If only one class, use placeholder values
            print("\nWARNING: Only one class present in test set. Some metrics cannot be calculated.")
            print("For proper evaluation, ensure test set contains both real and fake samples.\n")
            
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
            metrics['eer'] = 0.5
            metrics['eer_threshold'] = 0.5
            metrics['roc_auc'] = 0.5
            metrics['fpr'] = np.array([0, 1])
            metrics['tpr'] = np.array([0, 1])
        
        # Confusion matrix can be computed regardless
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        
        return metrics
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {model_path}")
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    def plot_history(self):
        """
        Plot training history.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        if 'val_acc' in self.history and len(self.history['val_acc']) > 0:
            ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, confusion_mat):
        """
        Plot confusion matrix.
        
        Args:
            confusion_mat: Confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        return plt.gcf()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: Area under the ROC curve
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        return plt.gcf() 