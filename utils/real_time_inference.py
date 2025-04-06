"""
Real-time audio deepfake detection using the AASIST model.
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import sounddevice as sd
import soundfile as sf
import threading
import queue
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.aasist_model import AASIST
from utils.data_processing import pad_or_truncate, extract_mel_spectrogram

class RealTimeAudioDeepfakeDetector:
    """
    Real-time audio deepfake detector.
    """
    def __init__(self, model_path, config):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to the trained model
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
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
        
        # Audio buffer
        self.audio_buffer = np.zeros(config['chunk_size'] * config['buffer_chunks'], dtype=np.float32)
        self.buffer_idx = 0
        
        # Processing queue
        self.queue = queue.Queue()
        
        # Results
        self.detection_results = []
        self.detection_probabilities = []
        
        # Create output directory if it doesn't exist
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        print(f"Model loaded from {model_path}")
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio input.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time: Time
            status: Status
        """
        if status:
            print(f"Status: {status}")
        
        # Add audio to buffer
        self.audio_buffer[self.buffer_idx:self.buffer_idx + len(indata)] = indata[:, 0]
        self.buffer_idx += len(indata)
        
        # If buffer is full, process it
        if self.buffer_idx >= len(self.audio_buffer):
            # Reset buffer index
            self.buffer_idx = 0
            
            # Copy buffer for processing
            audio_data = np.copy(self.audio_buffer)
            
            # Add to processing queue
            self.queue.put(audio_data)
    
    def process_audio(self):
        """
        Process audio data from the queue.
        """
        while True:
            # Get audio data from queue
            audio_data = self.queue.get()
            
            # Process audio
            fake_prob, prediction = self.predict_audio(audio_data)
            
            # Add to results
            self.detection_results.append(prediction)
            self.detection_probabilities.append(fake_prob)
            
            # Print result
            result = "FAKE" if prediction == 1 else "REAL"
            print(f"Prediction: {result}, Fake Probability: {fake_prob:.4f}")
            
            # Mark task as done
            self.queue.task_done()
    
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
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fake_prob = probabilities[0, 1].item()  # Probability of being fake
            prediction = 1 if fake_prob > 0.5 else 0  # 1 for fake, 0 for real
        
        return fake_prob, prediction
    
    def start_recording(self, duration=None):
        """
        Start recording and processing audio.
        
        Args:
            duration: Duration to record in seconds, None for infinite
        """
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start recording
        stream = sd.InputStream(
            channels=1,
            samplerate=self.config['sample_rate'],
            blocksize=self.config['chunk_size'],
            callback=self.audio_callback
        )
        
        print(f"Starting real-time audio deepfake detection (Press Ctrl+C to stop)")
        
        try:
            with stream:
                if duration is None:
                    # Record until interrupted
                    while True:
                        time.sleep(0.1)
                else:
                    # Record for specified duration
                    for _ in tqdm(range(int(duration * 10)), desc="Recording"):
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nRecording stopped")
        finally:
            # Wait for all tasks to be processed
            self.queue.join()
            
            # Save results
            if self.detection_results:
                # Calculate overall result
                fake_count = sum(self.detection_results)
                real_count = len(self.detection_results) - fake_count
                fake_percentage = fake_count / len(self.detection_results) * 100
                
                print(f"\nAnalysis complete:")
                print(f"Total segments: {len(self.detection_results)}")
                print(f"Real segments: {real_count} ({100 - fake_percentage:.1f}%)")
                print(f"Fake segments: {fake_count} ({fake_percentage:.1f}%)")
                
                # Final verdict
                if fake_percentage > 50:
                    print("\nVerdict: The audio is likely FAKE")
                else:
                    print("\nVerdict: The audio is likely REAL")
                
                # Save results to file
                result_file = os.path.join(self.config['output_dir'], 'detection_results.txt')
                with open(result_file, 'w') as f:
                    f.write(f"Total segments: {len(self.detection_results)}\n")
                    f.write(f"Real segments: {real_count} ({100 - fake_percentage:.1f}%)\n")
                    f.write(f"Fake segments: {fake_count} ({fake_percentage:.1f}%)\n")
                    
                    if fake_percentage > 50:
                        f.write("\nVerdict: The audio is likely FAKE\n")
                    else:
                        f.write("\nVerdict: The audio is likely REAL\n")
                    
                    f.write("\nDetailed results:\n")
                    for i, (result, prob) in enumerate(zip(self.detection_results, self.detection_probabilities)):
                        result_str = "FAKE" if result == 1 else "REAL"
                        f.write(f"Segment {i+1}: {result_str} (Fake Probability: {prob:.4f})\n")
                
                print(f"Results saved to {result_file}")
    
    def process_file(self, file_path):
        """
        Process an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        print(f"Processing file: {file_path}")
        
        # Load audio file
        audio, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Resample if needed
        if sr != self.config['sample_rate']:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sample_rate'])
        
        # Process audio in chunks
        chunk_size = int(self.config['duration'] * self.config['sample_rate'])
        overlap = int(chunk_size * 0.5)  # 50% overlap
        
        # Pad audio to ensure at least one full chunk
        if len(audio) < chunk_size:
            padded_audio = np.zeros(chunk_size)
            padded_audio[:len(audio)] = audio
            audio = padded_audio
        
        # Process chunks
        i = 0
        while i < len(audio) - chunk_size + 1:
            # Extract chunk
            chunk = audio[i:i+chunk_size]
            
            # Process chunk
            fake_prob, prediction = self.predict_audio(chunk)
            
            # Add to results
            self.detection_results.append(prediction)
            self.detection_probabilities.append(fake_prob)
            
            # Print result
            result = "FAKE" if prediction == 1 else "REAL"
            print(f"Chunk {len(self.detection_results)}: {result}, Fake Probability: {fake_prob:.4f}")
            
            # Move to next chunk
            i += overlap
        
        # Calculate overall result
        fake_count = sum(self.detection_results)
        real_count = len(self.detection_results) - fake_count
        fake_percentage = fake_count / len(self.detection_results) * 100
        
        print(f"\nAnalysis complete:")
        print(f"Total chunks: {len(self.detection_results)}")
        print(f"Real chunks: {real_count} ({100 - fake_percentage:.1f}%)")
        print(f"Fake chunks: {fake_count} ({fake_percentage:.1f}%)")
        
        # Final verdict
        if fake_percentage > 50:
            print("\nVerdict: The audio is likely FAKE")
        else:
            print("\nVerdict: The audio is likely REAL")
        
        # Save results to file
        result_file = os.path.join(self.config['output_dir'], f"{os.path.basename(file_path)}_results.txt")
        with open(result_file, 'w') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Total chunks: {len(self.detection_results)}\n")
            f.write(f"Real chunks: {real_count} ({100 - fake_percentage:.1f}%)\n")
            f.write(f"Fake chunks: {fake_count} ({fake_percentage:.1f}%)\n")
            
            if fake_percentage > 50:
                f.write("\nVerdict: The audio is likely FAKE\n")
            else:
                f.write("\nVerdict: The audio is likely REAL\n")
            
            f.write("\nDetailed results:\n")
            for i, (result, prob) in enumerate(zip(self.detection_results, self.detection_probabilities)):
                result_str = "FAKE" if result == 1 else "REAL"
                f.write(f"Chunk {i+1}: {result_str} (Fake Probability: {prob:.4f})\n")
        
        print(f"Results saved to {result_file}")

def main():
    """
    Main function for real-time audio deepfake detection.
    """
    parser = argparse.ArgumentParser(description='Real-time audio deepfake detection.')
    parser.add_argument('--model', type=str, default='../models/checkpoints/best_model.pth', help='Path to the trained model')
    parser.add_argument('--input_file', type=str, help='Path to input audio file (optional)')
    parser.add_argument('--duration', type=float, help='Duration to record in seconds (optional)')
    parser.add_argument('--output_dir', type=str, default='../results/real_time', help='Directory to save results')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--raw_input', action='store_true', help='Use raw waveform as input')
    parser.add_argument('--no_sinc_filter', action='store_false', dest='sinc_filter', help='Disable sinc filters')
    parser.add_argument('--no_graph_attention', action='store_false', dest='graph_attention', help='Disable graph attention')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'sample_rate': args.sample_rate,
        'duration': 4.0,  # Duration of each segment in seconds
        'chunk_size': 1024,  # Number of samples per chunk
        'buffer_chunks': 62,  # Number of chunks to buffer (to reach 'duration' seconds)
        'n_mels': 80,  # Number of Mel bands
        'raw_input': args.raw_input,
        'sinc_filter': args.sinc_filter,
        'graph_attention': args.graph_attention,
        'n_class': 2,  # Number of classes (real/fake)
        'output_dir': args.output_dir
    }
    
    # Create detector
    detector = RealTimeAudioDeepfakeDetector(args.model, config)
    
    # Process audio file or record
    if args.input_file:
        detector.process_file(args.input_file)
    else:
        detector.start_recording(args.duration)

if __name__ == '__main__':
    main() 