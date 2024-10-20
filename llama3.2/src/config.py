class Config:
    """
    Configuration class that holds the settings and hyperparameters used for training the model and setting up the environment.

    Attributes:
        OUTPUT_DIR (str): Path to the directory where the model will be saved after training.
        MODEL_ID (str): Identifier for the base model, which will be loaded from the Unsloth model repository.
        DATA_DIR (str): Path to the JSON data file used for training.
        MAX_SEQ_LENGTH (int): Maximum sequence length for tokenization and input sequences.
        BATCH_SIZE (int): Batch size per device, which depends on the available VRAM and the dataset size.
        MAX_STEP (int): Maximum number of training steps. This value usually corresponds to 3-10 epochs depending on dataset size.
        LEARNING_RATE (float): Learning rate for the AdamW optimizer.
        WEIGHT_DECAY (float): Weight decay to apply for the optimizer to prevent overfitting.
        GRADIENT_ACC_STEPS (int): Number of gradient accumulation steps. Gradients are accumulated over these steps before updating the model weights.
        WARM_UP_STEPS (int): Number of steps for learning rate warm-up, to avoid rapid changes at the start of training.
        LORA_RANK (int): Rank for Low-Rank Adaptation (LoRA). This is a hyperparameter that controls the rank of the low-rank matrices in the LoRA technique (e.g., 8, 16, 32, 64, 128).
        LORA_ALPHA (int): Scaling factor for LoRA. Typically set to 1x or 2x the value of `LORA_RANK` to control the update magnitude in the LoRA layers.
    """
    
    OUTPUT_DIR = "/root/machinelearning/llama3.2/llama3.2-3B-Instruct-test"
    MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
    DATA_DIR = "/root/machinelearning/llama3.2/data/about_me.json"
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 8  # Depends on VRAM and dataset
    MAX_STEP = 20  # 3-10 Epoches
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    GRADIENT_ACC_STEPS = 2
    WARM_UP_STEPS = 10
    LORA_RANK = 64  # 8, 16, 32, 64, 128
    LORA_ALPHA = 128  # Suggested 1x or 2x Lora rank
