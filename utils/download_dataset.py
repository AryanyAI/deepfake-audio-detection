"""
Script to download the ASVspoof 2019 dataset.
"""

import os
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm
import shutil

def download_file(url, save_path, chunk_size=8192):
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        save_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_file(file_path, extract_dir):
    """
    Extract a zip or tar file.
    
    Args:
        file_path: Path to the file to extract
        extract_dir: Directory to extract to
    """
    os.makedirs(extract_dir, exist_ok=True)
    
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif file_path.endswith('.tar'):
        with tarfile.open(file_path, 'r:') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def download_asvspoof_2019(data_dir):
    """
    Download the ASVspoof 2019 dataset.
    
    Args:
        data_dir: Directory to save the dataset
    """
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    print("The ASVspoof dataset URLs are no longer valid through direct download.")
    print("Please follow these steps to download the dataset manually:")
    print("1. Visit: https://zenodo.org/records/14498691")
    print("2. Download the dataset files you need (LA_T.zip, LA_D.zip, LA_E.zip)")
    print("3. Download the protocols: ASVspoof2019_LA_cm_protocols.zip")
    print(f"4. Place these files in: {data_dir}")
    print(f"5. Extract them manually to subfolders in: {data_dir}")
    print("\nAlternatively, you can download from the original source:")
    print("https://datashare.ed.ac.uk/handle/10283/3336")
    
    # Ask if user wants to proceed with manual download
    print("\nWould you like to open the Zenodo page in your browser? (yes/no)")
    response = input().lower()
    if response == 'yes':
        import webbrowser
        webbrowser.open('https://zenodo.org/records/14498691')

def download_asvspoof_sample(data_dir):
    """
    Download a small sample of the ASVspoof 2019 dataset for testing purposes.
    
    Args:
        data_dir: Directory to save the dataset
    """
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample directory structure
    sample_dir = os.path.join(data_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    
    print("The ASVspoof dataset URLs are no longer valid through direct download.")
    print("Please follow these steps to download the dataset manually:")
    print("1. Visit: https://zenodo.org/records/14498691")
    print("2. Download the following files:")
    print("   - LA_D.zip (Development set)")
    print("   - ASVspoof2019_LA_cm_protocols.zip (Protocols)")
    print(f"3. Place these files in: {data_dir}")
    print(f"4. Extract them to: {sample_dir}")
    print("\nAlternatively, you can download a smaller sample dataset from Kaggle:")
    print("https://www.kaggle.com/datasets/birdy654/asvspoof-2019")

def main():
    """
    Main function to download the ASVspoof 2019 dataset.
    """
    parser = argparse.ArgumentParser(description='Download the ASVspoof 2019 dataset.')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='Directory to save the dataset')
    parser.add_argument('--sample', action='store_true', help='Download only a small sample of the dataset')
    
    args = parser.parse_args()
    
    if args.sample:
        download_asvspoof_sample(args.data_dir)
    else:
        download_asvspoof_2019(args.data_dir)

if __name__ == '__main__':
    main() 