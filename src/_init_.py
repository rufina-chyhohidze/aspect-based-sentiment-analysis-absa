import kagglehub
import os
import shutil


def download_dataset():
    # Download dataset
    path = kagglehub.dataset_download("farukalam/yelp-restaurant-reviews")
    print("Downloaded to:", path)

    # Define your own data folder
    target_folder = "data"
    os.makedirs(target_folder, exist_ok=True)

    # Copy files recursively
    for root, dirs, files in os.walk(path):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(target_folder, file)
            shutil.copy(src, dst)

    print("Files saved to:", target_folder)
    print("Saved files:", os.listdir(target_folder))


if __name__ == '__main__':
    # Example: run your dataset download
    download_dataset()
