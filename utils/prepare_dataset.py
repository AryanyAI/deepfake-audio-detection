"""
Script to prepare the ASVspoof dataset after manual download.
"""

import os
import argparse
import zipfile
import tarfile
import shutil
from tqdm import tqdm

def extract_file(file_path, extract_dir):
    """
    Extract a zip or tar file.
    
    Args:
        file_path: Path to the file to extract
        extract_dir: Directory to extract to
    """
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {os.path.basename(file_path)} to {extract_dir}...")
    
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(file_path)}"):
                    zip_ref.extract(member, extract_dir)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                for member in tqdm(tar_ref.getmembers(), desc=f"Extracting {os.path.basename(file_path)}"):
                    tar_ref.extract(member, extract_dir)
        elif file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r:') as tar_ref:
                for member in tqdm(tar_ref.getmembers(), desc=f"Extracting {os.path.basename(file_path)}"):
                    tar_ref.extract(member, extract_dir)
        print(f"Successfully extracted {os.path.basename(file_path)}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False

def prepare_dataset(data_dir):
    """
    Prepare the ASVspoof dataset after manual download.
    
    Args:
        data_dir: Directory containing downloaded dataset files
    """
    # Define dataset subdirectories for ASVspoof5 (updated from ASVspoof2019)
    subsets = {
        'flac_T_': 'LA_train',
        'flac_D_': 'LA_dev',
        'flac_E_': 'LA_eval',
        'ASVspoof5_protocols': 'protocols',
        'LA_T': 'LA_train',
        'LA_D': 'LA_dev',
        'LA_E': 'LA_eval',
        'ASVspoof2019_LA_cm_protocols': 'protocols'
    }
    
    # Check if directories already exist (already extracted)
    required_dirs = ['LA_dev', 'protocols']
    already_extracted = True
    for required_dir in required_dirs:
        dir_path = os.path.join(data_dir, required_dir)
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            already_extracted = False
            break
    
    if already_extracted:
        print("Dataset directories already exist. Skipping extraction.")
        print("\nDataset preparation completed!")
        print("\nNext steps:")
        print("1. Run preprocessing: python utils/preprocess_data.py")
        print("2. Train the model: python notebooks/aasist_training.py")
        return
    
    # Check for downloaded files
    found_files = []
    for filename in os.listdir(data_dir):
        full_path = os.path.join(data_dir, filename)
        if os.path.isfile(full_path):
            # Match partial filenames
            for prefix, subdir in subsets.items():
                if filename.startswith(prefix):
                    found_files.append((filename, full_path, subdir))
                    break
    
    if not found_files:
        print("No dataset files found. Creating necessary directories for synthetic data.")
        # Create necessary directories
        for subdir in set(subsets.values()):
            os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
        return
    
    print(f"Found {len(found_files)} dataset files.")
    
    # Extract files
    for filename, file_path, extract_subdir in found_files:
        extract_dir = os.path.join(data_dir, extract_subdir)
        extract_file(file_path, extract_dir)
    
    print("\nDataset preparation completed!")
    print("\nNext steps:")
    print("1. Run preprocessing: python utils/preprocess_data.py")
    print("2. Train the model: python notebooks/aasist_training.py")

def main():
    """
    Main function to prepare the ASVspoof dataset.
    """
    parser = argparse.ArgumentParser(description='Prepare the ASVspoof dataset after manual download.')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='Directory containing downloaded dataset files')
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    
    prepare_dataset(args.data_dir)

if __name__ == '__main__':
    main() 