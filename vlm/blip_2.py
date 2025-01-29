import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor, AutoProcessor
from utils import load_and_convert_image


class BLIP2Evaluator:
    def __init__(self, model_id="Salesforce/blip2-opt-2.7b", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return load_and_convert_image(image_path)

    def evaluate(self, prompt, image_path, max_new_tokens=100):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f'BLIP-2: {generated_text}')
        return generated_text



if __name__ == "__main__":
    # Usage example
    evaluator = BLIP2Evaluator(device="cuda")  # Use "cuda" for GPU, or "cpu" for CPU
    result = evaluator.evaluate(prompt="Describe the image.", image_path="https://example.com/sample-image.jpg")
    print(result)

