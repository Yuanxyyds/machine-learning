class Config:
    """
    Config class for defining hyperparameters, data paths, and settings for a food classification model.

    Attributes:
        BATCH_SIZE (int): The number of samples per batch for training.
        N_CLASSES (int): The number of food categories to classify (20 in this case).
        IMAGE_WIDTH (int): The width of the input images (in pixels) used for training.
        IMAGE_HEIGHT (int): The height of the input images (in pixels) used for training.
        DATA_DIR (str): The directory path where the original images for the dataset are stored.
        TRAIN_DATA_METADATA (str): Path to the metadata file that lists the training data samples.
        TEST_DATA_METADATA (str): Path to the metadata file that lists the testing data samples.
        MINI_TRAIN_DATA_DIR (str): Path to the directory containing a smaller subset of training data (for faster testing or debugging).
        MINI_TEST_DATA_DIR (str): Path to the directory containing a smaller subset of testing data (for faster testing or debugging).
        TRAIN_DATA_DIR (str): Path to the processed training data directory.
        TEST_DATA_DIR (str): Path to the processed testing data directory.
        MODELS_DIR (str): Path to store trained models.
        LOGS_DIR (str): Path to store training logs.
        PLOTS_DIR (str): Path to store plot visualizations generated during the project.
        TUNER_DIR (str): Path to store models used during hyperparameter tuning.
        FOOD_LIST (list of str): List of food categories that the model will classify.
        TRAIN_SAMPLE_SIZE (int): The number of samples to be used for training. -1 indicates all available samples are used.
        TEST_SAMPLE_SIZE (int): The number of samples to be used for testing. -1 indicates all available samples are used.

    This configuration class centralizes important parameters and file paths for training, testing, and model management.
    It simplifies the workflow by providing a single source of truth for hyperparameters and dataset locations.
    """

    BATCH_SIZE = 32
    N_CLASSES = 20
    IMAGE_WIDTH = 299
    IMAGE_HEIGHT = 299
    DATA_DIR = "food-101/data/images/"
    TRAIN_DATA_METADATA = "food-101/data/meta/train.txt"
    TEST_DATA_METADATA = "food-101/data/meta/test.txt"
    MINI_TRAIN_DATA_DIR = "food-101/processed_data/mini_train"
    MINI_TEST_DATA_DIR = "food-101/processed_data/mini_test"
    TRAIN_DATA_DIR = "food-101/processed_data/train"
    TEST_DATA_DIR = "food-101/processed_data/test"
    MODELS_DIR = "food-101/models/"
    LOGS_DIR = "food-101/logs/"
    PLOTS_DIR = "food-101/plots/"
    FOOD_LIST = [
        "apple_pie",
        "baby_back_ribs",
        "bibimbap",
        "caesar_salad",
        "cheesecake",
        "chicken_curry",
        "chicken_wings",
        "club_sandwich",
        "donuts",
        "dumplings",
        "french_fries",
        "hot_dog",
        "pizza",
        "ramen",
        "steak",
        "ice_cream",
        "waffles",
        "spring_rolls",
        "sushi",
        "fish_and_chips",
    ]
    TRAIN_SAMPLE_SIZE = -1
    TEST_SAMPLE_SIZE = -1
