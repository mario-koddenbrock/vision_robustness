import os.path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import login


def get_model_size(model_id):
    """
    Get the size of a model on Hugging Face.

    Args:
        model_id (str): The ID of the model on Hugging Face.

    Returns:
        int: The size of the model in bytes.
    """
    api = HfApi()
    model_info = api.model_info(model_id)
    total_size = sum(file.size for file in model_info.siblings if file.size is not None)
    return total_size

def format_size(size_in_bytes):
    """
    Format the size from bytes to a human-readable format.

    Args:
        size_in_bytes (int): The size in bytes.

    Returns:
        str: The size in a human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024


def huggingface_login():
    """
    Load Huggingface access token from cache and login.

    Args:

    """

    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    access_token_HF = load_huggingface_token(cache_path)
    if access_token_HF:
        print(f"Access Token: {access_token_HF}")
        login(token=access_token_HF)
    else:
        print("Access Token not found in cache")


def load_huggingface_token(cache_path):
    """
    Load Huggingface access token from cache.

    Args:
        cache_path (str): Path to the cache file containing the token.

    Returns:
        str: The access token if found, otherwise None.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read().strip()
    return None

def check_model_in_cache(model_id):
    """
    Check if a model is already downloaded in the cache.

    Args:
        model_id (str): The ID of the model on Hugging Face.

    Returns:
        bool: True if the model is in the cache, False otherwise.
    """
    try:
        hf_hub_download(repo_id=model_id, filename="", local_dir=None, local_dir_use_symlinks=False)
        return True
    except:
        return False

if __name__ == "__main__":
    model_id = "bert-base-uncased"  # Replace with your model ID
    size_in_bytes = get_model_size(model_id)
    print(f"Model size: {format_size(size_in_bytes)}")