"""
Simplified inference script for audio deepfake detection using the AASIST model.
Optimized for systems with limited hardware resources.
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import soundfile as sf
import librosa
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.aasist_model import AASIST
from utils.data_processing import pad_or_truncate, extract_mel_spectrogram

class SimpleAudioDeepfakeDetector:
    """
    Simplified audio deepfake detector for systems with limited resources.
    """
    def __init__(self, model_path, config):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the trained model
            config: Configuration dictionary
        """
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config['force_cpu'] else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        print("Loading model...")
        self.model = AASIST(
            raw_input=config['raw_input'],
            sinc_filter=config['sinc_filter'],
            graph_attention=config['graph_attention'],
            n_class=config['n_class']
        )
        
        # Load model weights
        self.load_model(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            print(f"Model loaded successfully from {model_path}")
            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            sys.exit(1)
    
    def predict_audio(self, audio):
        """
        Predict whether audio contains real or fake speech.
        
        Args:
            audio: Audio signal
            
        Returns:
            Prediction probability and label
        """
        # Preprocess audio
        if self.config['raw_input']:
            # For raw waveform input
            audio = pad_or_truncate(audio, target_length=int(self.config['duration'] * self.config['sample_rate']))
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        else:
            # For spectrogram input
            audio = pad_or_truncate(audio, target_length=int(self.config['duration'] * self.config['sample_rate']))
            mel_spec = extract_mel_spectrogram(audio, sr=self.config['sample_rate'], n_mels=self.config['n_mels'])
            audio_tensor = torch.from_numpy(mel_spec).float().to(self.device)
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Perform inference
        with torch.no_grad():
            # Use smaller batch size or process in chunks if needed
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fake_prob = probabilities[0, 1].item()  # Probability of being fake
            prediction = 1 if fake_prob > 0.5 else 0  # 1 for fake, 0 for real
        
        return fake_prob, prediction
    
    def process_file(self, file_path):
        """
        Process an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        print(f"Processing file: {file_path}")
        
        try:
            # Load audio file
            audio, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Resample if needed
            if sr != self.config['sample_rate']:
                print(f"Resampling from {sr}Hz to {self.config['sample_rate']}Hz...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sample_rate'])
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return
        
        # Process in chunks to avoid memory issues
        chunk_size = int(self.config['duration'] * self.config['sample_rate'])
        overlap = int(chunk_size * 0.5)  # 50% overlap between chunks
        
        # Ensure we have at least one full chunk
        if len(audio) < chunk_size:
            padded_audio = np.zeros(chunk_size)
            padded_audio[:len(audio)] = audio
            audio = padded_audio
        
        # Process in overlapping chunks
        results = []
        probs = []
        i = 0
        
        with tqdm(total=(len(audio) - chunk_size) // (chunk_size - overlap) + 1, desc="Processing audio") as pbar:
            while i < len(audio) - chunk_size + 1:
                # Extract chunk
                chunk = audio[i:i+chunk_size]
                
                # Process chunk
                fake_prob, prediction = self.predict_audio(chunk)
                
                # Store results
                results.append(prediction)
                probs.append(fake_prob)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'chunk': len(results), 'fake_prob': f"{fake_prob:.4f}"})
                
                # Move to next position
                i += chunk_size - overlap
        
        # If no chunks were processed, return
        if not results:
            print("No results obtained. File may be too short.")
            return
        
        # Analyze results
        fake_count = sum(results)
        real_count = len(results) - fake_count
        fake_percentage = 100 * fake_count / len(results)
        avg_fake_prob = sum(probs) / len(probs)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Total chunks analyzed: {len(results)}")
        print(f"Real chunks: {real_count} ({100 - fake_percentage:.1f}%)")
        print(f"Fake chunks: {fake_count} ({fake_percentage:.1f}%)")
        print(f"Average fake probability: {avg_fake_prob:.4f}")
        
        # Final verdict
        verdict = "FAKE" if fake_percentage > 50 else "REAL"
        confidence = max(fake_percentage, 100 - fake_percentage)
        
        print(f"\nVerdict: The audio is likely {verdict} (Confidence: {confidence:.1f}%)")
        
        # Save results to file
        result_file = os.path.join(self.config['output_dir'], f"{os.path.basename(file_path)}_results.txt")
        with open(result_file, 'w') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Total chunks analyzed: {len(results)}\n")
            f.write(f"Real chunks: {real_count} ({100 - fake_percentage:.1f}%)\n")
            f.write(f"Fake chunks: {fake_count} ({fake_percentage:.1f}%)\n")
            f.write(f"Average fake probability: {avg_fake_prob:.4f}\n")
            f.write(f"\nVerdict: The audio is likely {verdict} (Confidence: {confidence:.1f}%)\n")
            
            f.write("\nDetailed results:\n")
            for i, (result, prob) in enumerate(zip(results, probs)):
                result_str = "FAKE" if result == 1 else "REAL"
                f.write(f"Chunk {i+1}: {result_str} (Fake Probability: {prob:.4f})\n")
        
        print(f"Results saved to {result_file}")
        
        return verdict, confidence

def main():
    """
    Main function for simplified audio deepfake detection.
    """
    parser = argparse.ArgumentParser(description='Simplified audio deepfake detection for limited hardware.')
    parser.add_argument('--model', type=str, default='./models/checkpoints/best_model.pth', help='Path to the trained model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--raw_input', action='store_true', help='Use raw waveform as input')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'sample_rate': args.sample_rate,
        'duration': 4.0,  # Duration of each segment in seconds
        'n_mels': 80,  # Number of Mel bands
        'raw_input': args.raw_input,
        'sinc_filter': True,
        'graph_attention': True,
        'n_class': 2,  # Number of classes (real/fake)
        'output_dir': args.output_dir,
        'force_cpu': args.force_cpu
    }
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
    
    # Create detector
    detector = SimpleAudioDeepfakeDetector(args.model, config)
    
    # Process file
    start_time = time.time()
    detector.process_file(args.input_file)
    processing_time = time.time() - start_time
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")

if __name__ == '__main__':
    main() 