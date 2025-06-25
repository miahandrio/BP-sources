from transformers import AutoTokenizer, LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM
import torch
from PIL import Image


class LlavaNextModel:
    def __init__(self, model_name="llava-hf/llava-v1.6-vicuna-13b-hf"):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
        
        # Move the model to the desired device (e.g., GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        
        # Default processor
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


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