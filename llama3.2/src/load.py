from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from config import Config
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template


def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_ID,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3", 
    )
    return model, tokenizer


def load_data():
    # Load the local JSON file
    dataset = load_dataset(
        "json",
        data_files="/root/machinelearning/llama3.2/data/about_me.json",
        split="train",
    )
    return dataset



def tokenize_data(tokenizer, data):
    def mapping(data):
        texts = [tokenizer.apply_chat_template(example, tokenize = False, add_generation_prompt = False) for example in data["input"]]
        return { "text" : texts, }
    return data.map(mapping, batched=True)


def fine_tune_model(tokenized_data, model, tokenizer):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_data,
        dataset_text_field="text",
        max_seq_length=Config.MAX_SEQ_LENGTH,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACC_STEPS,
            warmup_steps=Config.WARM_UP_STEPS,
            max_steps=Config.MAX_STEP,
            learning_rate=Config.LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir=Config.OUTPUT_DIR,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=3407,
            lr_scheduler_type="linear",
        ),
    )
    trainer.train()
    model.save_pretrained(f"{Config.OUTPUT_DIR}/model", tokenizer, save_method = "merged_16bit",)


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    data = load_data()
    tokenized_data = tokenize_data(data=data, tokenizer=tokenizer)
    fine_tune_model(tokenized_data, model, tokenizer)
