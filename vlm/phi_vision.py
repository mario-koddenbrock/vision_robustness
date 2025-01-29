from transformers import AutoModelForCausalLM, AutoProcessor

from utils import load_and_convert_image


class PhiVisionEvaluator:
    def __init__(self, model_id="microsoft/Phi-3.5-vision-instruct", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        # Load the model with specific configurations and move to the appropriate device (CPU/GPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='eager',
            low_cpu_mem_usage=True,
        ).to(self.device)

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, num_crops=4)

    def _load_image(self, image_path):
        return load_and_convert_image(image_path)

    def evaluate(self, prompt, image_path):
        # Load the image (from URL or local)
        image = self._load_image(image_path)

        # Prepare inputs for the model
        machine_prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = self.processor(machine_prompt, image, return_tensors="pt").to(self.device)

        # Generate the response
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=1000, eos_token_id=self.processor.tokenizer.eos_token_id
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response


if __name__ == "__main__":
    # Usage example
    evaluator = PhiVisionEvaluator(device="cuda")  # For GPU, or "cpu" for CPU
    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg")
    print(result)

    result = evaluator.evaluate(image_path="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg")
    print(result)
