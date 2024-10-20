from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from config import Config
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template


class ModelAdaptorTrainer:
    """
    A class to handle the setup, training, and saving of a LoRA (Low-Rank Adaptation) adaptor model using the Unsloth library,
    Hugging Face transformers, and other tools. This class includes methods to generate the LoRA adaptor, load data, tokenize it,
    and train the model.

    Methods:
        run(): Orchestrates the process of generating the adaptor, loading the dataset, tokenizing, and training the model.
        generate_adaptor_and_tokenizer(): Initializes the base language model and LoRA adaptor using Unsloth's
                                          FastLanguageModel, along with a tokenizer.
        load_data(): Loads the dataset from a local JSON file using Hugging Face's `load_dataset`.
        tokenize_data(dataset, tokenizer): Tokenizes the dataset by applying a chat template to each input example.
        train_adaptor(lora_adaptor, tokenizer, tokens): Trains the LoRA adaptor model using the specified training arguments
                                                        and dataset, then saves the model.
    """

    def run(self):
        """
        Main method to run the entire process, including generating the adaptor and tokenizer, loading the dataset,
        tokenizing the data, and training the model.
        """
        lora_adaptor, tokenizer = self.generate_adaptor_and_tokenizer()
        dataset = self.load_data()
        tokens = self.tokenize_data(dataset, tokenizer)
        self.train_adaptor(lora_adaptor, tokenizer, tokens)

    def generate_adaptor_and_tokenizer(self):
        """
        Generates the LoRA adaptor model and the tokenizer. The LoRA adaptor is fine-tuned based on the base model
        using the specified configuration for LoRA tuning.

        Returns:
            tuple: The LoRA adaptor model and the tokenizer.
        """
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=Config.MODEL_ID,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )

        lora_adaptor = FastLanguageModel.get_peft_model(
            base_model,
            r=Config.LORA_RANK,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            use_rslora=False,
            loftq_config=None,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3",
        )
        return lora_adaptor, tokenizer

    def load_data(self):
        """
        Loads the training data from a specified local JSON file. The file path is configured via the `Config.DATA_DIR`.

        Returns:
            Dataset: The loaded dataset in Hugging Face's `Dataset` format.
        """
        return load_dataset(
            "json",
            data_files=Config.DATA_DIR,
            split="train",
        )

    def tokenize_data(self, dataset, tokenizer):
        """
        Tokenizes the dataset by applying the tokenizer's chat template to the input examples.

        Args:
            dataset (Dataset): The dataset to tokenize.
            tokenizer (Tokenizer): The tokenizer that applies the chat template.

        Returns:
            Dataset: The tokenized dataset.
        """

        def mapping(data):
            texts = [
                tokenizer.apply_chat_template(
                    example, tokenize=False, add_generation_prompt=False
                )
                for example in data["input"]
            ]
            return {
                "text": texts,
            }

        return dataset.map(mapping, batched=True)

    def train_adaptor(self, lora_adaptor, tokenizer, tokens):
        """
        Trains the LoRA adaptor model using the specified training dataset and tokenizer. Configures the training
        process through Hugging Face's `TrainingArguments` and the `SFTTrainer` for supervised fine-tuning (SFT).
        After training, the model is saved to the output directory.

        Args:
            lora_adaptor (Model): The LoRA adaptor model to be trained.
            tokenizer (Tokenizer): The tokenizer used for processing the dataset.
            tokens (Dataset): The tokenized dataset to be used for training.
        """
        trainer = SFTTrainer(
            model=lora_adaptor,
            tokenizer=tokenizer,
            train_dataset=tokens,
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
                weight_decay=Config.WEIGHT_DECAY,
                seed=3407,
                lr_scheduler_type="linear",
            ),
        )
        trainer.train()
        lora_adaptor.save_pretrained(
            f"{Config.OUTPUT_DIR}/adaptor",
            tokenizer,
            save_method="merged_16bit",
        )


if __name__ == "__main__":
    """
    Main script execution: Creates an instance of the ModelAdaptorTrainer class and starts the training process by calling
    the `run()` method.
    """
    adaptor_trainer = ModelAdaptorTrainer()
    adaptor_trainer.run()
