from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import Config

class ModelMerger:
    """
    A class to handle the merging of a PEFT (Parameter-Efficient Fine-Tuning) model with its base model. 
    This results in a merged model that combines the fine-tuned parameters with the baseline model, 
    which can then be saved and used for inference without the need for separate PEFT adapters.

    Methods:
        merge_adapot_with_baseline(): Merges the fine-tuned PEFT model with the base model and saves the merged model 
                                      along with the tokenizer to the specified output directory.
    """

    def merge_adapot_with_baseline(self):
        """
        Merges the fine-tuned LoRA (PEFT) adapter with the base model, unloads the adapter to consolidate the parameters, 
        and saves the fully merged model and tokenizer to the output directory.
        """
        base_model = AutoModelForCausalLM.from_pretrained(Config.MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
        peft_model_id = f"{Config.OUTPUT_DIR}/model"
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        merged_model = model.merge_and_unload()
        
        # Save the merged model and tokenizer to the specified output directory
        merged_model.save_pretrained(f"{Config.OUTPUT_DIR}/full-model")
        tokenizer.save_pretrained(f"{Config.OUTPUT_DIR}/full-model")


if __name__ == "__main__":
    """
    Main script execution: Instantiates the ModelMerger class and calls the method to merge and save the model.
    """
    modelMerger = ModelMerger()
    modelMerger.merge_adapot_with_baseline()
