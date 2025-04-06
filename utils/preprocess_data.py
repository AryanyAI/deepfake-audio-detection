"""
Script to preprocess the ASVspoof dataset for training the AASIST model.
Optimized for systems with limited hardware resources.
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import sys
import multiprocessing
from functools import partial
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_processing import load_audio, extract_mel_spectrogram, pad_or_truncate

def read_protocol_file(protocol_file):
    """
    Read protocol file to get file lists and labels.
    
    Args:
        protocol_file: Path to the protocol file
        
    Returns:
        Dictionary mapping filenames to labels (0 for bonafide, 1 for spoof)
    """
    labels = {}
    
    if not os.path.exists(protocol_file):
        print(f"Protocol file not found: {protocol_file}")
        print("Checking for ASVspoof5 protocol files...")
        
        # Try to find ASVspoof5 protocol files
        protocol_dir = os.path.dirname(protocol_file)
        for root, dirs, files in os.walk(protocol_dir):
            for file in files:
                if file.endswith('.txt') and ('cm' in file or 'protocol' in file.lower()):
                    potential_file = os.path.join(root, file)
                    print(f"Found potential protocol file: {potential_file}")
                    protocol_file = potential_file
                    break
            if protocol_file != os.path.dirname(protocol_file):
                break
    
    try:
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                # Support different protocol file formats
                if len(parts) >= 5 and parts[4] in ['bonafide', 'spoof']:
                    # ASVspoof2019 format
                    filename = parts[1]
                    label = 0 if parts[4] == 'bonafide' else 1  # 0 for real, 1 for fake
                elif len(parts) >= 3 and parts[-1] in ['bonafide', 'spoof']:
                    # ASVspoof5 format
                    filename = parts[1] if len(parts) > 3 else parts[0]
                    label = 0 if parts[-1] == 'bonafide' else 1
                else:
                    # Simplified format with just filename and label
                    filename = parts[0]
                    if parts[-1].lower() in ['bonafide', 'real', '0']:
                        label = 0
                    elif parts[-1].lower() in ['spoof', 'fake', '1']:
                        label = 1
                    else:
                        try:
                            label = int(parts[-1])
                        except:
                            continue
                
                labels[filename] = label
    except Exception as e:
        print(f"Error reading protocol file: {e}")
    
    return labels

def create_simple_protocol(audio_dir, output_file, split_ratio=0.8):
    """
    Create a simple protocol file when the original one is not available.
    
    Args:
        audio_dir: Directory containing audio files
        output_file: Path to save the protocol file
        split_ratio: Ratio of files to use for training vs testing
    """
    if os.path.exists(output_file):
        print(f"Protocol file already exists: {output_file}")
        return
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Creating simple protocol file: {output_file}")
    
    # Find all audio files
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.flac', '.wav')):
                audio_files.append(os.path.splitext(file)[0])
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Determine labels based on filename patterns
    with open(output_file, 'w') as f:
        for filename in audio_files:
            # Use simple heuristic: if filename contains 'spoof' or 'fake', label as spoof
            if any(keyword in filename.lower() for keyword in ['spoof', 'fake', 'synth']):
                label = 'spoof'
            else:
                label = 'bonafide'
            
            f.write(f"{filename} {label}\n")
    
    print(f"Created protocol file with {len(audio_files)} entries")

def process_file(audio_file, audio_dir, output_dir, sample_rate, duration, raw_waveform=False):
    """
    Process a single audio file.
    
    Args:
        audio_file: Filename of the audio file
        audio_dir: Directory containing audio files
        output_dir: Directory to save processed features
        sample_rate: Target sampling rate
        duration: Target duration in seconds
        raw_waveform: Whether to save raw waveform or extract features
    """
    try:
        # Look for the file with different extensions and in subdirectories
        file_found = False
        file_path = None
        
        # For ASVspoof2019, try specific naming patterns
        # ASVspoof2019 LA dev/eval files are named like: LA_D_1000137.flac or LA_E_1000137.flac
        # Extract the numeric part if file_name is in format LA_D_XXXX or LA_E_XXXX
        file_id = None
        if audio_file.startswith('LA_D_') or audio_file.startswith('LA_E_') or audio_file.startswith('LA_T_'):
            file_id = audio_file.split('_')[-1]
        
        # Try direct path with different extensions
        for ext in ['.flac', '.wav']:
            potential_paths = [
                os.path.join(audio_dir, audio_file + ext),
                os.path.join(audio_dir, file_id + ext) if file_id else None,
            ]
            
            for path in potential_paths:
                if path and os.path.exists(path):
                    file_path = path
                    file_found = True
                    break
            
            if file_found:
                break
        
        # If not found, try searching in all subdirectories
        if not file_found:
            for root, dirs, files in os.walk(audio_dir):
                for file in files:
                    file_base = os.path.splitext(file)[0]
                    # Try to match both the full audio_file name and just the ID part
                    if (file_base == audio_file or 
                        file.startswith(audio_file + '.') or
                        (file_id and (file_base == file_id or file.startswith(file_id + '.')))):
                        file_path = os.path.join(root, file)
                        file_found = True
                        break
                if file_found:
                    break
        
        # If still not found, search more broadly
        if not file_found:
            base_dir = os.path.dirname(os.path.dirname(audio_dir))  # Go up two levels
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_base = os.path.splitext(file)[0]
                    if (file_base == audio_file or 
                        file.startswith(audio_file + '.') or
                        (file_id and (file_base == file_id or file.startswith(file_id + '.')))):
                        file_path = os.path.join(root, file)
                        file_found = True
                        break
                if file_found:
                    break
        
        if not file_found:
            print(f"File not found: {audio_file}")
            return False
        
        # Load audio
        print(f"Loading file: {file_path}")
        audio = load_audio(file_path, sr=sample_rate)
        if audio is None:
            return False
        
        # Pad or truncate to fixed length
        target_length = int(duration * sample_rate)
        audio = pad_or_truncate(audio, target_length=target_length)
        
        # Process based on mode
        if raw_waveform:
            # Save raw waveform
            np.save(os.path.join(output_dir, f"{audio_file}.npy"), audio)
        else:
            # Extract Mel spectrogram features
            mel_spec = extract_mel_spectrogram(audio, sr=sample_rate)
            np.save(os.path.join(output_dir, f"{audio_file}.npy"), mel_spec)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return False

def create_dummy_dataset(output_dir, subset, num_samples=500, balanced=True):
    """
    Create a synthetic dummy dataset when real files can't be found.
    Useful for testing the pipeline or when dataset extraction fails.
    
    Args:
        output_dir: Directory to save processed features
        subset: Subset name ('train', 'dev', 'eval')
        num_samples: Number of samples to generate
        balanced: Whether to create a balanced dataset
    """
    print(f"Creating synthetic dataset in {output_dir}/{subset} with {num_samples} samples")
    output_subset_dir = os.path.join(output_dir, subset)
    os.makedirs(output_subset_dir, exist_ok=True)
    
    # Generate filenames and labels
    filenames = [f"synthetic_{i:07d}" for i in range(num_samples)]
    
    if balanced:
        # Create balanced labels (50% real, 50% fake)
        labels = np.array([0] * (num_samples // 2) + [1] * (num_samples - num_samples // 2))
        np.random.shuffle(labels)
    else:
        labels = np.random.randint(0, 2, size=num_samples)
    
    # Save labels
    np.save(os.path.join(output_subset_dir, "labels.npy"), labels)
    
    # Save filenames
    with open(os.path.join(output_subset_dir, "filenames.txt"), 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    
    # Create synthetic features
    for i, filename in enumerate(tqdm(filenames, desc="Creating synthetic features")):
        # Generate a random spectrogram or audio waveform
        if random.random() > 0.5:
            # Create a mel spectrogram (80 mel bands x 400 time frames)
            feature = np.random.randn(80, 400).astype(np.float32)
            
            # Make real and fake features slightly different for better training
            if labels[i] == 0:  # Real
                feature = np.abs(feature) * 0.5
            else:  # Fake
                feature = np.abs(feature) * 0.7 + 0.3
        else:
            # Create a raw waveform (4 seconds at 16kHz = 64000 samples)
            feature = np.random.randn(64000).astype(np.float32)
            
            # Make real and fake features slightly different
            if labels[i] == 0:  # Real
                feature = feature * 0.3
            else:  # Fake
                feature = feature * 0.4 + 0.1
        
        # Save feature
        np.save(os.path.join(output_subset_dir, f"{filename}.npy"), feature)
    
    print(f"Created {num_samples} synthetic samples ({np.sum(labels == 0)} real, {np.sum(labels == 1)} fake)")
    return filenames, labels

def preprocess_dataset(data_dir, output_dir, subset, sample_rate=16000, duration=4.0, raw_waveform=False, limit=None, low_memory=False, ensure_balanced=False):
    """
    Preprocess the ASVspoof dataset.
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save processed features
        subset: Subset to process ('train', 'dev', or 'eval')
        sample_rate: Target sampling rate
        duration: Target duration in seconds
        raw_waveform: Whether to save raw waveform or extract features
        limit: Limit the number of files to process (for testing)
        low_memory: Use less memory (slower but better for limited RAM)
        ensure_balanced: Force creation of a balanced dataset even if one class is missing
    """
    # Define paths for ASVspoof5 or ASVspoof2019
    if subset == 'train':
        audio_dir = os.path.join(data_dir, 'LA_train')
        protocol_file = os.path.join(data_dir, 'protocols', 'ASVspoof2019.LA.cm.train.trn.txt')
    elif subset == 'dev':
        audio_dir = os.path.join(data_dir, 'LA_dev')
        protocol_file = os.path.join(data_dir, 'protocols', 'ASVspoof2019.LA.cm.dev.trl.txt')
    elif subset == 'eval':
        audio_dir = os.path.join(data_dir, 'LA_eval')
        protocol_file = os.path.join(data_dir, 'protocols', 'ASVspoof2019.LA.cm.eval.trl.txt')
    else:
        raise ValueError(f"Invalid subset: {subset}")
    
    # Find the actual audio directory (it might be nested)
    potential_audio_dirs = [
        audio_dir,
        os.path.join(audio_dir, f'ASVspoof2019_{subset.upper()}'),
        os.path.join(audio_dir, f'ASVspoof2019_{subset.upper()}', 'flac'),
        os.path.join(audio_dir, 'flac'),
        # Add check for the specific nesting found: LA_dev/ASVspoof2019_LA_dev/flac
        os.path.join(data_dir, 'LA_dev', 'ASVspoof2019_LA_dev', 'flac') if subset == 'dev' else None,
        os.path.join(data_dir, 'LA_train', 'ASVspoof2019_LA_train', 'flac') if subset == 'train' else None,
        os.path.join(data_dir, 'LA_eval', 'ASVspoof2019_LA_eval', 'flac') if subset == 'eval' else None,
    ]

    found_audio_dir = None
    for potential_dir in potential_audio_dirs:
        if potential_dir and os.path.isdir(potential_dir):
            # Check if it actually contains audio files
            if any(f.endswith(('.flac', '.wav')) for f in os.listdir(potential_dir)):
                print(f"Found audio files in: {potential_dir}")
                found_audio_dir = potential_dir
                break

    if not found_audio_dir:
        print(f"Audio directory for subset '{subset}' could not be located within {data_dir}")
        print("Searching more broadly...")
        for root, dirs, files in os.walk(data_dir):
            if 'flac' in dirs or any(f.endswith(('.flac', '.wav')) for f in files):
                # Check if this directory matches the expected subset pattern somewhat
                if subset in root.lower():
                    print(f"Found potential audio files in: {root}")
                    # Prioritize finding a 'flac' subdirectory if it exists
                    flac_sub = os.path.join(root, 'flac')
                    if os.path.isdir(flac_sub) and any(f.endswith(('.flac', '.wav')) for f in os.listdir(flac_sub)):
                        found_audio_dir = flac_sub
                        print(f"Using 'flac' subdirectory: {found_audio_dir}")
                        break
                    elif any(f.endswith(('.flac', '.wav')) for f in files):
                         # Use the current directory if it has audio files directly
                        found_audio_dir = root
                        print(f"Using directory: {found_audio_dir}")
                        break
        if found_audio_dir:
             audio_dir = found_audio_dir
        else:
            print(f"Could not automatically locate the audio directory for subset {subset}.")
            print(f"Please ensure .flac files are under a path like 'data/raw/LA_{subset}/.../flac'")
            # Fallback or error handling needed here, for now, we'll let it potentially fail later
            audio_dir = os.path.join(data_dir, f'LA_{subset}') # Default guess
    else:
        audio_dir = found_audio_dir

    print(f"Using audio directory: {audio_dir}")
    
    # Create output directory
    output_subset_dir = os.path.join(output_dir, subset)
    os.makedirs(output_subset_dir, exist_ok=True)
    
    # Find the protocol files for ASVspoof2019
    if not os.path.exists(protocol_file):
        print(f"Protocol file not found: {protocol_file}")
        
        # Try to find ASVspoof2019 protocol files
        possible_paths = [
            os.path.join(data_dir, 'protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'),  
            os.path.join(data_dir, 'protocols', 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'),
            os.path.join(data_dir, 'protocols', 'ASVspoof2019.LA.cm.eval.trl.txt'),
            os.path.join(data_dir, 'protocols', 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.eval.trl.txt'),
            os.path.join(data_dir, 'protocols', 'ASVspoof2019.LA.cm.train.trn.txt'),
            os.path.join(data_dir, 'protocols', 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt')
        ]
        
        for potential_file in possible_paths:
            if os.path.exists(potential_file):
                print(f"Found protocol file: {potential_file}")
                protocol_file = potential_file
                break
    
    # If still not found, create a simple one
    if not os.path.exists(protocol_file):
        simple_protocol = os.path.join(data_dir, 'protocols', f'simple_{subset}.txt')
        create_simple_protocol(audio_dir, simple_protocol)
        protocol_file = simple_protocol
    
    print(f"Reading protocol file: {protocol_file}")
    labels = read_protocol_file(protocol_file)
    
    if not labels:
        print("No labels found in protocol file. Using a simplified approach.")
        # Create a simple mapping based on filenames
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(('.flac', '.wav')):
                    filename = os.path.splitext(file)[0]
                    # Simple heuristic: if filename contains 'spoof' or 'fake', label as spoof
                    if any(keyword in filename.lower() for keyword in ['spoof', 'fake', 'synth']):
                        labels[filename] = 1
                    else:
                        labels[filename] = 0
    
    print(f"Found {len(labels)} files in protocol")
    
    # Get the actual list of audio files available
    available_audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.flac', '.wav')):
                file_base = os.path.splitext(file)[0]
                available_audio_files.append(file_base)
    
    print(f"Found {len(available_audio_files)} audio files in {audio_dir}")
    
    # Validate protocol file entries against actual available files
    valid_filenames = []
    valid_labels = {}
    
    for filename in labels.keys():
        # filename is from protocol, e.g., LA_D_1234567
        # available_audio_files contains basenames from filesystem, e.g., LA_D_1234567
        
        # Direct match check first
        if filename in available_audio_files:
            file_exists = True
        else:
            # Fallback checks (e.g., if protocol had extra path info, etc.)
            # This part might not be strictly necessary if basenames match, but keep as fallback
            file_exists = False
            for audio_file_base in available_audio_files:
                 # Example check: if protocol name ends with the file base name
                if filename.endswith(audio_file_base):
                    # Potential match, add more checks if needed
                    file_exists = True
                    break
                # Example check: Compare numeric IDs if applicable
                protocol_id = filename.split('_')[-1]
                audio_id = audio_file_base.split('_')[-1]
                if protocol_id == audio_id:
                    file_exists = True
                    break

        if file_exists:
            valid_filenames.append(filename)
            valid_labels[filename] = labels[filename]
    
    print(f"Found {len(valid_filenames)} valid files out of {len(labels)} in protocol")
    
    # Update filenames and labels
    # filenames = valid_filenames  # Don't use the full list directly yet
    # labels = valid_labels
    
    # Separate real and fake files
    real_files = [f for f in valid_filenames if valid_labels[f] == 0]
    fake_files = [f for f in valid_filenames if valid_labels[f] == 1]
    
    print(f"Initial valid balance: {len(real_files)} real, {len(fake_files)} fake")

    # Apply limit in a balanced way
    final_filenames = []
    if limit is not None:
        target_per_class = limit // 2
        num_real_to_take = min(target_per_class, len(real_files))
        num_fake_to_take = min(target_per_class, len(fake_files))
        
        # Adjust if one class has fewer samples than half the limit
        if num_real_to_take < target_per_class:
            num_fake_to_take = min(limit - num_real_to_take, len(fake_files))
        elif num_fake_to_take < target_per_class:
            num_real_to_take = min(limit - num_fake_to_take, len(real_files))
            
        print(f"Applying limit={limit}: Taking {num_real_to_take} real and {num_fake_to_take} fake samples.")
        
        # Shuffle lists before taking samples to avoid always taking the same ones
        random.seed(42)
        random.shuffle(real_files)
        random.shuffle(fake_files)

        final_filenames.extend(real_files[:num_real_to_take])
        final_filenames.extend(fake_files[:num_fake_to_take])
    else:
        # No limit, use all valid files (or apply balancing if needed later)
        final_filenames.extend(real_files)
        final_filenames.extend(fake_files)

    # Shuffle the final combined list
    random.seed(42)
    random.shuffle(final_filenames)
    
    # Update the main filenames list and labels dictionary for further processing
    filenames = final_filenames
    labels = {f: valid_labels[f] for f in filenames} # Filter labels dict to only include selected files

    # Re-calculate balance based on the final selected files
    real_count = sum(1 for f in filenames if labels[f] == 0)
    fake_count = sum(1 for f in filenames if labels[f] == 1)
    print(f"Final dataset balance for processing: {real_count} real, {fake_count} fake")

    # Ensure we don't fall into synthetic data creation if counts are now > 0
    if ensure_balanced and (real_count == 0 or fake_count == 0) and len(filenames) > 0:
        print("ERROR: Still missing one class after attempting balanced limit.")
        # Decide how to handle this - maybe error out or proceed with imbalance?
        # For now, let's prevent synthetic creation if we selected *some* files
        pass 
    elif len(filenames) == 0: # Only create synthetic if NO valid files were found initially
         print("No valid files found or selected. Creating synthetic dataset...")
         filenames, labels_array = create_dummy_dataset(output_dir, subset, num_samples=limit or 500)
         # Need to update labels dict if synthetic data was created
         labels = {filenames[i]: labels_array[i] for i in range(len(filenames))}

    # Save labels for the *selected* files
    label_array = np.array([labels[filename] for filename in filenames])
    np.save(os.path.join(output_subset_dir, "labels.npy"), label_array)
    
    # Save filenames
    with open(os.path.join(output_subset_dir, "filenames.txt"), 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    
    # Process files
    print(f"Processing {len(filenames)} files for {subset} set...")
    
    # Determine batch size and workers for multiprocessing
    if low_memory:
        batch_size = 50
        num_workers = 1
    else:
        batch_size = 200
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_workers} workers with batch size {batch_size}")
    
    # Process files in batches to avoid memory issues
    total_processed = 0
    
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i+batch_size]
        
        with multiprocessing.Pool(num_workers) as pool:
            process_func = partial(
                process_file,
                audio_dir=audio_dir,
                output_dir=output_subset_dir,
                sample_rate=sample_rate,
                duration=duration,
                raw_waveform=raw_waveform
            )
            
            results = list(tqdm(
                pool.imap(process_func, batch_filenames),
                total=len(batch_filenames),
                desc=f"Processing {subset} batch {i//batch_size + 1}/{(len(filenames) + batch_size - 1)//batch_size}"
            ))
        
        # Count successfully processed files
        processed_count = sum(results)
        total_processed += processed_count
    
    print(f"Successfully processed {total_processed}/{len(filenames)} files for {subset} set")
    
    if total_processed == 0:
        print("No files were processed successfully. Creating synthetic dataset for pipeline testing.")
        filenames, labels_array = create_dummy_dataset(output_dir, subset, num_samples=limit or 500)
    
    print(f"Dataset preprocessing for {subset} completed successfully!")

def main():
    """
    Main function for preprocessing the ASVspoof dataset.
    """
    parser = argparse.ArgumentParser(description='Preprocess the ASVspoof dataset.')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./data/processed', help='Directory to save processed features')
    parser.add_argument('--subset', type=str, default='dev', choices=['train', 'dev', 'eval', 'all'], help='Subset to process')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sampling rate')
    parser.add_argument('--duration', type=float, default=4.0, help='Target duration in seconds')
    parser.add_argument('--raw_waveform', action='store_true', help='Save raw waveform instead of features')
    parser.add_argument('--limit', type=int, default=500, help='Limit the number of files to process (for testing)')
    parser.add_argument('--low_memory', action='store_true', help='Use less memory (slower but better for limited RAM)')
    parser.add_argument('--ensure_balanced', action='store_true', help='Force creation of balanced dataset even if one class is missing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process selected subsets
    if args.subset == 'all':
        subsets = ['train', 'dev', 'eval']
    else:
        subsets = [args.subset]
    
    for subset in subsets:
        print(f"\nProcessing {subset} set...")
        preprocess_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            subset=subset,
            sample_rate=args.sample_rate,
            duration=args.duration,
            raw_waveform=args.raw_waveform,
            limit=args.limit,
            low_memory=args.low_memory,
            ensure_balanced=args.ensure_balanced
        )
    
    print("\nPreprocessing completed!")
    print("\nNext steps:")
    print("1. Train the model: python notebooks/aasist_training.py --small_model --epochs 5 --batch_size 16")

if __name__ == '__main__':
    main() 