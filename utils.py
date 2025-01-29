import os.path
import time

import pandas as pd
import requests
from PIL import Image

import os
import shutil

def load_and_convert_image(image_path):
    """
    Load an image from a web URL or local path and convert it to RGB format.

    Args:
        image_path (str): Path to the image (can be a URL or local path).

    Returns:
        Image: The loaded and converted image.
    """
    if image_path.startswith('http://') or image_path.startswith('https://'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        image = Image.open(image_path)
    else:
        raise ValueError(f"Invalid image path: {image_path}")

    # Ensure the image is in RGB format (fixes issues with incorrect color channels)
    return image.convert("RGB")


def initialize_evaluators(device, config):
    """
    Initialize evaluators based on the provided configuration.

    Args:
        device (str): The device to run the evaluators on (e.g., 'cpu' or 'cuda').
        config (dict): Configuration dictionary mapping evaluator names to their classes.

    Returns:
        dict: A dictionary of initialized evaluators.
    """
    evaluators = {}
    for name, evaluator_class in config.items():
        start_time = time.time()
        evaluators[name] = evaluator_class(device=device)
        end_time = time.time()
        print(f"Initialized {name} in {end_time - start_time:.2f} seconds")
    return evaluators




def reorganize_dataset(dataset_dir, good_class_name="good", output_dir="datasets/reorganized_dataset"):
    # Create output directories for 'good' and 'defective'
    good_dir = os.path.join(output_dir, 'good')
    defective_dir = os.path.join(output_dir, 'defective')

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(defective_dir, exist_ok=True)

    # Iterate over train and test directories
    for split in ['train', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Determine if the class should be 'good' or 'defective'
            if class_name == good_class_name:
                dest_dir = good_dir
            else:
                dest_dir = defective_dir

            # Move and rename images
            for img_name in os.listdir(class_dir):
                src_path = os.path.join(class_dir, img_name)
                if os.path.isfile(src_path):
                    # Rename the image to include class name and split (train/test)
                    new_name = f"{class_name}_{split}_{img_name}"
                    dest_path = os.path.join(dest_dir, new_name)
                    shutil.copy2(src_path, dest_path)  # Copy with metadata

    print(f"Dataset reorganized into: {output_dir}")


def reorganize_csv_dataset(dataset_dir, output_dir="datasets/reorganized_dataset"):
    # Paths to CSV files
    train_csv = os.path.join(dataset_dir, "Training_set.csv")
    test_csv = os.path.join(dataset_dir, "Testing_set.csv")

    # Paths to train and test image directories
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Function to move files based on the csv file
    def process_split(csv_file, split, src_dir):
        # Load CSV
        data = pd.read_csv(csv_file)

        # Assume the CSV has columns 'image' and 'class'
        for _, row in data.iterrows():
            img_name = row['filename']
            class_name = row['label']

            # Create class folder if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Source and destination paths
            src_path = os.path.join(src_dir, img_name)
            if os.path.isfile(src_path):
                new_name = f"{class_name}_{split}_{img_name}"
                dest_path = os.path.join(class_dir, new_name)
                shutil.copy2(src_path, dest_path)  # Copy with metadata

    # Process training set
    if os.path.exists(train_csv) and os.path.exists(train_dir):
        process_split(train_csv, "train", train_dir)

    # # Process testing set
    # if os.path.exists(test_csv) and os.path.exists(test_dir):
    #     process_split(test_csv, "test", test_dir)

    print(f"Dataset reorganized into: {output_dir}")


import os
import shutil


def keep_first_n_images(main_directory, n=20):
    # Traverse the main directory
    for subdir, _, files in os.walk(main_directory):
        # Only process subdirectories (ignoring the main directory itself)
        if subdir != main_directory:
            # Sort the files (you can sort by name or any other criteria)
            files.sort()

            # Check if there are more than n files
            if len(files) > n:
                # Select the files that should be deleted (those beyond the first n)
                files_to_delete = files[n:]

                # Delete each file
                for file_name in files_to_delete:
                    file_path = os.path.join(subdir, file_name)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")


# example usage
# if __name__ == "__main__":
#     # reorganize_dataset('datasets/leather')
#     # reorganize_dataset('datasets/zipper')
#     # reorganize_csv_dataset('datasets/Human Action Recognition')
# 
#
#     keep_first_n_images('datasets/Food-101', n=20)
#     keep_first_n_images('datasets/Human Action Recognition', n=20)
#     keep_first_n_images('datasets/Human Emotions', n=20)
#     keep_first_n_images('datasets/Satellite Image Classification', n=20)
#     keep_first_n_images('datasets/leather', n=20)
#     keep_first_n_images('datasets/zipper', n=20)


