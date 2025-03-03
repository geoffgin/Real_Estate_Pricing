import os
import kaggle

def download_kaggle_dataset(dataset, path):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
    print(f"Downloaded dataset {dataset} to {path}")

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory {dir_path}")
    else:
        print(f"Directory {dir_path} already exists")

# Kaggle dataset path
dataset_path = "new-york-city/nyc-property-sales"

# Local directory to save the dataset
local_dir = "./assets"

# Create the 'assets' directory if it doesn't exist
create_directory(local_dir)

# Download the dataset to the 'assets' directory
download_kaggle_dataset(dataset_path, local_dir)