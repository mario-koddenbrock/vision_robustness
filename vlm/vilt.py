import torch
from transformers import ViltModel, ViltProcessor, ViltForQuestionAnswering, ViltForMaskedLM

from utils import load_and_convert_image


class ViltEvaluator:
    def __init__(self, model_id="dandelin/vilt-b32-mlm", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        self.model = ViltForMaskedLM.from_pretrained(self.model_id)
        self.processor = ViltProcessor.from_pretrained(self.model_id)


    def _load_image(self, image_path):
        return load_and_convert_image(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        encoding = self.processor(image, prompt, return_tensors="pt")
        outputs = self.model(**encoding)

        logits = outputs.logits
        idx = torch.sigmoid(logits).argmax(-1).item()

        answer = self.model.config.id2label[idx]
        print("Predicted answer:", answer)

        return answer