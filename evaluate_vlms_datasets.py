import warnings

import torch

from datasets import DATASET_DESCRIPTIONS, evaluate_dataset
from utils import initialize_evaluators
from vlm.SigLIP import SigLIPEvaluator
from vlm.blip_2 import BLIP2Evaluator
from vlm.blip import BLIPEvaluator
from vlm.clip import CLIPEvaluator
from vlm.pali_gemma import PaliGemmaEvaluator
from vlm.phi_vision import PhiVisionEvaluator
from vlm.pixtral import PixtralVisionEvaluator
from vlm.qwen_vl import QwenVLEvaluator
from vlm.vilt import ViltEvaluator
from vlm.visual_bert import VisualBertEvaluator

import huggingface as hf

# Ignore warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Log in to Hugging Face
    hf.huggingface_login()

    # Configuration for evaluators
    EVALUATORS_CONFIG = {
        "PhiVision": PhiVisionEvaluator,
        "SigLIP": SigLIPEvaluator,
        "Pixtral": PixtralVisionEvaluator,
        "CLIP": CLIPEvaluator,
        "PaliGemma": PaliGemmaEvaluator,
        # "QwenVL": QwenVLEvaluator,
        # "Vilt": ViltEvaluator,
        # "VisualBert": VisualBertEvaluator,
        # "BLIP": BLIPEvaluator,
        # "BLIP2": BLIP2Evaluator,
    }

    # Initialize evaluators
    evaluators = initialize_evaluators(device, EVALUATORS_CONFIG)


    datasets = [
        "Satellite Image Classification",
        "Human Action Recognition",
        "leather",
        "FER2013",
        "Food-101",
        "zipper",
    ]

    # Iterate over each dataset and evaluate
    for dataset_name in datasets:
        evaluate_dataset(evaluators, dataset_name)