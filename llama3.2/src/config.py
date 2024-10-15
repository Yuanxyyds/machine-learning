class Config:
    """
    TODO:
    """

    OUTPUT_DIR = "/root/machinelearning/llama3.2/llama3.2-3B-Instruct"
    MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 16
    MAX_STEP = 80
    LEARNING_RATE = 2e-3
    GRADIENT_ACC_STEPS = 4
    WARM_UP_STEPS = 5
