import os
import argparse
import subprocess

def download_and_unzip(dataset: str, path: str):
    # Make sure Kaggle API credentials are set
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("Kaggle API credentials not found. Place kaggle.json in ~/.kaggle/")
        return

    # Download the dataset using kaggle CLI
    subprocess.run(f"kaggle datasets download -p {path} {dataset}", shell=True)

        # Extract the dataset name from the identifier
    dataset_name = dataset.split("/")[-1]

    # Unzip the dataset to 'dataset' folder
    unzip_folder = os.path.join(path, dataset_name)
    os.makedirs(unzip_folder, exist_ok=True)
    subprocess.run(f"unzip {path}/*.zip -d {unzip_folder}", shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unzip Kaggle dataset.')
    parser.add_argument('dataset', type=str, help='Identifier for Kaggle dataset, in format "user/dataset_name"')
    parser.add_argument('--path', type=str, default='.', help='Path to download the dataset')
    args = parser.parse_args()

    download_and_unzip(args.dataset, args.path)
