import os
import random
from zipfile import ZipFile
import time
import requests
from medmnist import TissueMNIST, BloodMNIST

import plotting
import prompts
from vlm.clip import CLIPEvaluator
from vlm.visual_bert import VisualBertEvaluator


DATASET_DESCRIPTIONS = {
    "FER2013": "Images of faces for classification into emotional categories.",
    "Food-101": "Images of 101 different food categories for classification.",
    "Human Action Recognition": "Dataset of images of people performing various actions for classification.",
    "leather": "Image dataset for classifying the condition (defective or good) of leather patches.",
    "zipper": "Image dataset for classifying the condition (defective or good) of zippers.",
    "Human Emotions": "Images of people expressing different emotions for classification.",
    # "Aerial Drone Urban Classification": "Dataset of ~1800 Aerial Photographs of an Urban area classified into 5 categories, depending on which was most prevalent in the image",
    # "BloodMNIST": "Blood cell images for classification tasks from the MedMNIST collection.",
    # "BreakHis": "Histopathological images of breast cancer for benign and malignant classification.",
    # "EuroSAT": "Satellite images for land cover classification.",
    # "MVTec_AD": "Images for detecting anomalies and defects in industrial products.",
    # "PlantVillage": "Images for classifying plant diseases in various crops.",
    # "TissueMNIST": "Images of human tissue for classification tasks from the MedMNIST collection.",
}

DATASET_PATH = {
    "FER2013": "datasets/FER2013",
    "Food-101": "datasets/Food-101/images",
    "Human Action Recognition": "datasets/Human Action Recognition",
    "leather": "datasets/leather",
    "zipper": "datasets/zipper",
    "Satellite Image Classification": "datasets/Satellite Image Classification",
    "Human Emotions": "datasets/Human Emotions",
}

DATASET_URLS = {
    "FER2013": "TODO: Implement this",
    "Food-101": "TODO: Implement this",
    "Human Action Recognition": "TODO: Implement this",
    "leather": "TODO: Implement this",
    "zipper": "TODO: Implement this", # TODO: Implement this
    "Human Emotions": "TODO: Implement this",
    # "BloodMNIST": "https://medmnist.com/",
    # "BreakHis": "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images",
    # "EuroSAT": "https://zenodo.org/records/7711810#.ZAm3k-zMKEA",
    # "MVTec_AD": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
    # "PlantVillage": "https://www.kaggle.com/datasets/emmarex/plantdisease",
    # "TissueMNIST": "https://medmnist.com/",
}


def get_mvtec_ad():
    raise NotImplementedError("MVTec_AD dataset is not available for download. Please download it manually.")

def get_blood_mnist():
    path = "/Users/koddenbrock/Repository/vlm_robustness/datasets/BloodMNIST/"
    if not os.path.exists(path):
        os.makedirs(path)

    test_dataset = BloodMNIST(split="test", download=True, size=224, root=path)
    return test_dataset

def get_tissue_mnist():
    path = "/Users/koddenbrock/Repository/vlm_robustness/datasets/TissueMNIST/"
    if not os.path.exists(path):
        os.makedirs(path)

    test_dataset = TissueMNIST(split="test", download=True, size=224, root=path)
    return test_dataset


def get_breakhis():
    raise NotImplementedError("BreakHis dataset is not available for download. Please download it manually.")

def get_plantvillage():
    raise NotImplementedError("PlantVillage dataset is not available for download. Please download it manually.")

def get_fer2013():
    raise NotImplementedError("Fer2013 dataset is not available for download. Please download it manually.")

def get_eurosat():
    raise NotImplementedError("EuroSAT dataset is not available for download. Please download it manually.")

def get_food_101():
    raise NotImplementedError("FOOD-101 dataset is not available for download. Please download it manually.")



# Create a folder and download datasets
def download_dataset(name, url):
    # Ensure datasets/ directory exists
    base_folder = "datasets"
    os.makedirs(base_folder, exist_ok=True)

    # Create a folder for the dataset inside datasets/
    folder_name = os.path.join(base_folder, name.replace(" ", "_"))  # Replace spaces with underscores
    os.makedirs(folder_name, exist_ok=True)

    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Save the content as a zip file, assuming the datasets are zips
            zip_path = os.path.join(folder_name, f"{name}.zip")
            with open(zip_path, 'wb') as file:
                file.write(response.content)

            # Extract if it's a zip file
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder_name)

            print(f"{name} dataset downloaded and extracted successfully.")
        else:
            print(f"Failed to download {name}. URL might require special access.")
    except Exception as e:
        print(f"Error downloading {name}: {str(e)}")


def get_dataset_classes(path):
    folder = os.listdir(path)

    # if folder contains test and train folders, only use test
    if "test" in folder:
        folder = os.path.join(path, "test")

    # if folder contains images, only use images
    if "images" in folder:
        folder = os.path.join(path, "images")

    # filter out the folders that are not classes or are not visible
    classes = [f for f in folder if not f.startswith(".")]

    return classes


def get_class_files(dataset_path, class_name):

    classes = os.listdir(dataset_path)

    # if folder contains test and train folders, only use test
    if "test" in classes:
        dataset_path = os.path.join(dataset_path, "test")

    # if folder contains images, only use images
    if "images" in classes:
        dataset_path = os.path.join(dataset_path, "images")

    # get all images in dataset_path/class_name
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.exists(class_path):
        raise ValueError(f"Class {class_name} not found in {dataset_name} dataset.")

    class_subfolders = os.listdir(class_path)

    # if folder contains test and train folders, only use test
    if "test" in class_subfolders:
        class_path = os.path.join(class_path, "test")


    #get full file path to the images inside class_path
    files = [os.path.join(class_path, f) for f in os.listdir(class_path) if not f.startswith(".")]

    return files

def evaluate_image(evaluators, image_path):
    """
    Evaluate an image using the provided evaluators.

    Args:
        evaluators (dict): Dictionary of initialized evaluators.
        image_path (str): Path to the image to evaluate.
    """
    sample_prompt = "Describe the image:"

    results = {name: "default_result" for name in evaluators.keys()}
    for name, evaluator in evaluators.items():
        start_time = time.time()
        results[name] = evaluator.evaluate(prompt=sample_prompt, image_path=image_path)
        end_time = time.time()
        print(f"{name} Output: {results[name]} (Evaluated in {end_time - start_time:.2f} seconds)")

    image_name = os.path.basename(image_path)
    plotting.show_prediction_result(image_path, image_name, "Unknown", results)


def evaluate_dataset(evaluators, dataset_name, num_samples=1):
    """
    Evaluate a dataset using the provided evaluators.

    Args:
        evaluators (dict): Dictionary of initialized evaluators.
        dataset_name (str): Name of the dataset to evaluate.
    """

    dataset_description = DATASET_DESCRIPTIONS.get(dataset_name, "No description available.")
    dataset_path = DATASET_PATH.get(dataset_name, DATASET_PATH.get(dataset_name))

    classes = get_dataset_classes(dataset_path)
    if not classes:
        print(f"No classes found for dataset {dataset_path}")
        return

    print(f"Dataset: {dataset_name}")
    dataset_classes = get_dataset_classes(dataset_path)
    sample_prompt = prompts.classification_prompt(dataset_name, dataset_description, dataset_classes)

    for class_idx, class_name in enumerate(classes):
        print(f"\tClass: {class_name}")
        results = {name: "default_result" for name in evaluators.keys()}
        class_files = get_class_files(dataset_path, class_name)

        # shuffle the class files in random order
        random.shuffle(class_files)

        for sample_image_url in class_files[:num_samples]:
            print(f"\t\tSample image: {sample_image_url}")

            for name, evaluator in evaluators.items():

                # if the evaluator has a method set_class_names, set the class names
                set_class_names_op = getattr(evaluator, "set_class_names", None)
                if callable(set_class_names_op):
                    evaluator.set_class_names(classes)

                start_time = time.time()
                results[name] = evaluator.evaluate(prompt=sample_prompt, image_path=sample_image_url)
                end_time = time.time()
                print(f"\t\t{name} Output: {results[name]} ({end_time - start_time:.2f} sec.)")

            plotting.show_prediction_result(sample_image_url, dataset_name, class_name, results)

        # break after second class
        if class_idx == 2:
            break


if __name__ == "__main__":
    # Iterate over datasets and download them into the datasets/ directory
    for dataset_name, dataset_url in DATASET_URLS.items():

        # print(f"Downloading {dataset_name}...")
        # download_dataset(dataset_name, dataset_url)

        print(f"Dataset name: {dataset_name}")
        dataset_names = get_dataset_classes(dataset_name)

        print(f"Dataset classes: {dataset_names}")




