from transformers import BlipProcessor, BlipForConditionalGeneration

from utils import load_and_convert_image


class BLIPEvaluator:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        self.processor = BlipProcessor.from_pretrained(self.model_id)


    def _load_image(self, image_path):
        return load_and_convert_image(image_path)

    def evaluate(self, prompt, image_path, max_new_tokens=50):
        # Load the image (from URL or local)
        image = self._load_image(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)