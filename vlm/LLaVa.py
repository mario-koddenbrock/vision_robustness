from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration

from utils import load_and_convert_image


class LLaVaEvaluator:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        # Load the model and processor, and move the model to the appropriate device (CPU or GPU)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _load_image(self, image_path):
        return load_and_convert_image(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        prompts = [
            f"USER: <image>\nprompt\nASSISTANT:",
        ]

        inputs = self.processor(prompts, images=image, padding=True, return_tensors="pt").to("cuda")

        output = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)
        for text in generated_text:
            print(text.split("ASSISTANT:")[-1])

        generated_text = generated_text[0].split("ASSISTANT:")[-1].strip()

        return generated_text


if __name__ == "__main__":
    # Usage:
    evaluator = LlamaEvaluator()
    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg")
    print(result)

    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")
    print(result)
