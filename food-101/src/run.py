"""
Main entry point for running various stages of the food classification pipeline.

This script provides functionality for:
    - Preparing the system by checking GPU initialization and hardware configuration.
    - Loading and splitting the dataset into training and testing sets.
    - Visualizing the dataset by displaying images from different food categories.
    - Generating data for training and validation using data augmentation.
    - Training and fine-tuning deep learning models, including both baseline and transfer learning approaches.
    - Running predictions using a pre-trained model on a new image.
    - Comparing and showing the summary plots for each models 

Modules:
    - config: Contains configuration settings like dataset paths, model hyperparameters, etc.
    - model: Handles model architectures and training logic.
    - data_loader: Provides functions for visualizing and splitting the dataset.
    - prepare: Contains system preparation and hardware check routines.
    - train: Handles the model training process, including both baseline and fine-tuning methods.
    - result: Shows the summary plots for the models

Instructions:
    - Uncomment the relevant lines in the `if __name__ == "__main__":` block to run different parts of the pipeline.
    - For example, uncommenting `train.base_line_empty_model()` will start training the baseline model.
    - To run a prediction, use the `predict()` function with the appropriate model path and image path.

Usage:
    python main.py  # Uncomment and modify sections based on the desired functionality.
"""

import config, model, data_loader, prepare, train, result

if __name__ == "__main__":
    # prepare.check_gpu_init()
    # prepare.hardware_exam()
    # data_loader.visualize_dataset()
    # data_loader.split_train_and_test_data()
    # data_loader.split_mini_train_and_test_data()
    # gen, gen_v = train.data_generator()
    # train.base_line_empty_model(gen, gen_v)
    # train.fine_tune_vgg_model(gen, gen_v)
    # train.fine_tune_inception_model(gen, gen_v)
    # train.fine_tune_resnet_model(gen, gen_v)
    model.predict("food-101/models/22class32batch100epochs/baseline_model_22_class.keras", "food-101/website_data/apple_pie_1.jpg")
    model.predict("food-101/models/22class32batch100epochs/vgg_model_22_class.keras", "food-101/website_data/apple_pie_1.jpg")
    model.predict("food-101/models/22class32batch100epochs/inception_model_22_class.keras", "food-101/website_data/apple_pie_1.jpg")
    model.predict("food-101/models/22class32batch100epochs/resnet_model_22_class.keras", "food-101/website_data/apple_pie_1.jpg")
    model.predict("food-101/models/22class32batch100epochs/baseline_model_22_class.keras", "food-101/website_data/ramen1.jpg")    
    model.predict("food-101/models/22class32batch100epochs/vgg_model_22_class.keras", "food-101/website_data/ramen1.jpg")
    model.predict("food-101/models/22class32batch100epochs/inception_model_22_class.keras", "food-101/website_data/ramen1.jpg")
    model.predict("food-101/models/22class32batch100epochs/resnet_model_22_class.keras", "food-101/website_data/ramen1.jpg")
    model.predict("food-101/models/22class32batch100epochs/baseline_model_22_class.keras", "food-101/website_data/fish_and_chips1.jpeg")
    model.predict("food-101/models/22class32batch100epochs/vgg_model_22_class.keras", "food-101/website_data/fish_and_chips1.jpeg")
    model.predict("food-101/models/22class32batch100epochs/inception_model_22_class.keras", "food-101/website_data/fish_and_chips1.jpeg")
    model.predict("food-101/models/22class32batch100epochs/resnet_model_22_class.keras", "food-101/website_data/fish_and_chips1.jpeg")
    result.plot_accuracy(
        [
            "food-101/logs/22class32batch100epochs/baseline_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/vgg_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/inception_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/resnet_model_22_class.csv",
        ],
        ["Baseline", "VGG", "InceptionV3", "ResNet152"],
        "val_accuracy",
        "Validation Accuracy",
    )
    result.plot_accuracy(
        [
            "food-101/logs/22class32batch100epochs/baseline_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/vgg_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/inception_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/resnet_model_22_class.csv",
        ],
        ["Baseline", "VGG", "InceptionV3", "ResNet152"],
        "val_loss",
        "Validation Loss",
    )
    result.plot_accuracy(
        [
            "food-101/logs/22class32batch100epochs/baseline_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/vgg_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/inception_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/resnet_model_22_class.csv",
        ],
        ["Baseline", "VGG", "InceptionV3", "ResNet152"],
        "accuracy",
        "Accuracy",
    )
    result.plot_accuracy(
        [
            "food-101/logs/22class32batch100epochs/baseline_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/vgg_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/inception_model_22_class.csv",
            "food-101/logs/22class32batch100epochs/resnet_model_22_class.csv",
        ],
        ["Baseline", "VGG", "InceptionV3", "ResNet152"],
        "loss",
        "Loss",
    )
    pass
