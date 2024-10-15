from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import Config

base_model = AutoModelForCausalLM.from_pretrained(Config.MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
peft_model_id = "/root/machinelearning/llama3.2/llama3.2-3B-Instruct/model"
model = PeftModel.from_pretrained(base_model, peft_model_id)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{Config.OUTPUT_DIR}/full-model")
tokenizer.save_pretrained(f"{Config.OUTPUT_DIR}/full-model")