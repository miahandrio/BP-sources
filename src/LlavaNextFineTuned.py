from transformers import AutoTokenizer, AutoProcessor, LlavaNextForConditionalGeneration, LlavaForConditionalGeneration, AutoModelForCausalLM
import torch
from PIL import Image
from peft import PeftModel, PeftConfig

class LlavaNextFineTuned:
    def __init__(self, model_id="llava_lora_only-hledani-1e-5_1-epoch"):

        config = PeftConfig.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(config.base_model_name_or_path,
                                                             torch_dtype=torch.float16,
                                                                device_map="cuda")
        self.model = PeftModel.from_pretrained(self.model, model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        # Default processor
        self.processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        

    def evaluate(self, path, prompt):
        image = Image.open(path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # Extract assistant response only (after "ASSISTANT:")
        assistant_marker = "ASSISTANT:"
        if assistant_marker in response_full:
            response = response_full.split(assistant_marker)[-1].strip()
        else:
            response = response_full.strip()
        return response
