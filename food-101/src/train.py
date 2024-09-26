"""
This script provides functionality for training and fine-tuning deep learning models on the Food-101 dataset using Keras.

Functions:
    - data_generator(): Creates data generators for training and validation using data augmentation with Keras' ImageDataGenerator.
    - base_line_empty_model(generator, validation_generator): Defines a baseline convolutional neural network (CNN) model and trains it using the provided data generators.
    - fine_tune_vgg_model(generator, validation_generator): Fine-tunes a pre-trained VGG19 model using Keras Tuner for hyperparameter optimization and trains the model with the best hyperparameters.
    - fine_tune_inception_model(generator, validation_generator): Fine-tunes an InceptionV3 pre-trained model using Keras Tuner for hyperparameter optimization.
    - fine_tune_resnet_model(generator, validation_generator): Fine-tunes an ResNet152V2 pre-trained model using Keras Tuner for hyperparameter optimization.

Dependencies:
    - keras.api.models.Sequential: For building a sequential neural network model.
    - keras.api.applications.VGG19: Pre-trained VGG19 model used for transfer learning.
    - keras.api.applications.inception_v3.InceptionV3: Pre-trained InceptionV3 model used for transfer learning.
    - keras.api.layers: Common layers like Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAvgPool2D for building the model architecture.
    - keras.src.legacy.preprocessing.image.ImageDataGenerator: For data augmentation and loading image data.
    - keras.api.callbacks.CSVLogger, EarlyStopping: Callbacks for logging training metrics and implementing early stopping.
    - keras_tuner as kt: For hyperparameter tuning using Keras Tuner.
    - config.Config: Imports dataset configuration, including file paths, batch sizes, image dimensions, and food categories.

Usage:
    1. The `data_generator()` function should be called first to create training and validation data generators.
    2. You can train a baseline CNN model using `base_line_empty_model()`, which will save the trained model and log the training process.
    3. To perform transfer learning with hyperparameter tuning, use `fine_tune_vgg_model()`, which will search for the best hyperparameters and save the best vgg model.
    4. To perform transfer learning with hyperparameter tuning, use `fine_tune_inception_model()`, which will search for the best hyperparameters and save the best inception model.
    5. To perform transfer learning with hyperparameter tuning, use `fine_tune_resnet_model()`, which will search for the best hyperparameters and save the best resnet model.

Ensure the configuration in `config.Config` is correctly set before running the functions.

"""

import os
from keras.api.models import Sequential
from keras.api.applications import VGG19, ResNet152V2
from keras.api.applications.inception_v3 import InceptionV3
from keras.api.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAvgPool2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import CSVLogger, EarlyStopping
import keras_tuner as kt

from config import Config


def data_generator():
    """
    Creates data generators for training and validation data using data augmentation.

    The `ImageDataGenerator` class is used to apply various transformations (rescaling, rotation, shifting, zoom, etc.)
    to the images. The data is split into training and validation sets.

    Returns:
        tuple: A tuple containing two generators:
            - generator: The training data generator.
            - validation_generator: The validation data generator.
    """
    # Data Augmentation using ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,  # Split the data into training and validation
    )

    # Use the same generator for both training and validation
    generator = datagen.flow_from_directory(
        Config.MINI_TRAIN_DATA_DIR,
        target_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode="categorical",
        subset="training",  # For training data
        classes=Config.FOOD_LIST,
    )

    validation_generator = datagen.flow_from_directory(
        Config.MINI_TEST_DATA_DIR,
        target_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode="categorical",
        subset="validation",  # For validation data
        classes=Config.FOOD_LIST,
    )
    return generator, validation_generator


def base_line_empty_model(generator, validation_generator):
    """
    Builds and trains a baseline convolutional neural network (CNN) model.

    The model consists of several Conv2D layers, followed by MaxPooling, Flatten, Dense, and Dropout layers.
    The model is compiled with the Adam optimizer and categorical cross-entropy loss.

    Args:
        generator (ImageDataGenerator): The training data generator.
        validation_generator (ImageDataGenerator): The validation data generator.

    Returns:
        history: The training history of the model.
    """
    # Make the log and models folder if is not included
    if not os.path.exists(Config.LOGS_DIR):
        os.makedirs(Config.LOGS_DIR)
    if not os.path.exists(Config.MODELS_DIR):
        os.makedirs(Config.MODELS_DIR)

    # Create a sequential model
    model = Sequential()

    # Add convolutional layers with activation and pooling
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3),
        )
    )
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))

    # Flatten the output and add dense layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(Config.N_CLASSES, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    csv_logger = CSVLogger(
        f"{Config.LOGS_DIR}baseline_model_{Config.N_CLASSES}_class.log"
    )

    # Train the model with early stopping
    history = model.fit(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, csv_logger],
    )

    model.save(f"{Config.MODELS_DIR}baseline_model_{Config.N_CLASSES}_class.keras")
    return history


def fine_tune_vgg_model(generator, validation_generator):
    """
    Fine-tunes a VGG19 pre-trained model using Keras Tuner for hyperparameter optimization.

    This function freezes the VGG19 base model and adds custom Dense and Dropout layers.
    The Keras Tuner is used to search for the best hyperparameters, after which the best model is trained.

    Args:
        generator (ImageDataGenerator): The training data generator.
        validation_generator (ImageDataGenerator): The validation data generator.

    Returns:
        history_best_model: The training history of the best model.
    """
    # Make the log and models folder if is not included
    if not os.path.exists(Config.LOGS_DIR):
        os.makedirs(Config.LOGS_DIR)
    if not os.path.exists(Config.MODELS_DIR):
        os.makedirs(Config.MODELS_DIR)

    # Instantiate the VGG16 base model
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3),
    )
    base_model.trainable = False

    # Define a model-building function
    def build_model(hp):
        base_model = VGG19(
            weights="imagenet",
            include_top=False,
            input_shape=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3),
        )
        base_model.trainable = False

        model = Sequential(
            [
                base_model,
                Flatten(),
                Dense(256, activation="relu"),
                Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)),
                Dense(Config.N_CLASSES, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    # Instantiate the tuner
    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=Config.EPOCHS,
        factor=3,
        directory=f"{Config.MODELS_DIR}vggturner/",
        project_name="food_classification",
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Search for the best hyperparameters
    tuner.search(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping],
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters:\n{best_hps.values}")

    # Build the model with the best hyperparameters
    vgg_model = tuner.hypermodel.build(best_hps)

    csv_logger = CSVLogger(f"{Config.LOGS_DIR}vgg_model_{Config.N_CLASSES}_class.log")

    # Train the best model with the best hyperparameters
    history_vgg_model = vgg_model.fit(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, csv_logger],
    )

    # Save the best model and its weights
    vgg_model.save(f"{Config.MODELS_DIR}vgg_model_{Config.N_CLASSES}_class.keras")
    return history_vgg_model


def fine_tune_inception_model(generator, validation_generator):
    """
    Fine-tunes an InceptionV3 pre-trained model using Keras Tuner for hyperparameter optimization.

    This function freezes the InceptionV3 base model and adds custom Dense and Dropout layers.
    The Keras Tuner is used to search for the best hyperparameters, after which the best model is trained.

    Args:
        generator (ImageDataGenerator): The training data generator.
        validation_generator (ImageDataGenerator): The validation data generator.

    Returns:
        history_best_model: The training history of the best model.
    """
    # Make the log and models folder if not present
    if not os.path.exists(Config.LOGS_DIR):
        os.makedirs(Config.LOGS_DIR)
    if not os.path.exists(Config.MODELS_DIR):
        os.makedirs(Config.MODELS_DIR)

    inception_model = InceptionV3(
        weights="imagenet",
        include_top=False,  # Remove the fully connected layers at the top
        input_shape=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3),
    )
    inception_model.trainable = False  # Freeze the base model layers

    # Define a model-building function
    def build_model(hp):
        model = Sequential(
            [
                inception_model,
                GlobalAvgPool2D(),  # InceptionV3 typically uses this instead of Flatten
                Dense(256, activation="relu"),  # Add a fully connected layer
                Dropout(
                    hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)
                ),  # Tune the dropout rate
                Dense(Config.N_CLASSES, activation="softmax"),  # Output layer
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    # Instantiate the tuner
    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=Config.EPOCHS,
        factor=3,
        directory=f"{Config.MODELS_DIR}inceptionturner/",
        project_name="food_classification_inception",
    )

    # Early Stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Search for the best hyperparameters
    tuner.search(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping],
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters:\n{best_hps.values}")

    # Build the best model with the best hyperparameters
    inception_model = tuner.hypermodel.build(best_hps)

    csv_logger = CSVLogger(
        f"{Config.LOGS_DIR}inception_model_{Config.N_CLASSES}_class.log"
    )
    # Train the best model
    history_inception_model = inception_model.fit(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, csv_logger],
    )

    # Save the best model and weights
    inception_model.save(
        f"{Config.MODELS_DIR}inception_model_{Config.N_CLASSES}_class.keras"
    )
    return history_inception_model


def fine_tune_resnet_model(generator, validation_generator):
    """
    Fine-tunes a ResNet50 pre-trained model using Keras Tuner for hyperparameter optimization.

    This function freezes the ResNet50 base model and adds custom Dense and Dropout layers.
    The Keras Tuner is used to search for the best hyperparameters, after which the best model is trained.

    Args:
        generator (ImageDataGenerator): The training data generator.
        validation_generator (ImageDataGenerator): The validation data generator.

    Returns:
        history_best_model: The training history of the best model.
    """
    # Make the log and models folder if not present
    if not os.path.exists(Config.LOGS_DIR):
        os.makedirs(Config.LOGS_DIR)
    if not os.path.exists(Config.MODELS_DIR):
        os.makedirs(Config.MODELS_DIR)

    # Load the ResNet152 base model pre-trained on ImageNet
    resnet_model = ResNet152V2(
        weights="imagenet",
        include_top=False,  # Remove the fully connected layers at the top
        input_shape=(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 3),
    )
    resnet_model.trainable = False  # Freeze the base model layers

    # Define a model-building function
    def build_model(hp):
        model = Sequential(
            [
                resnet_model,
                GlobalAvgPool2D(),  # ResNet typically uses this instead of Flatten
                Dense(256, activation="relu"),  # Add a fully connected layer
                Dropout(
                    hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)
                ),  # Tune the dropout rate
                Dense(
                    Config.N_CLASSES, activation="softmax"
                ),  # Output layer for classification
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    # Instantiate the tuner
    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=Config.EPOCHS,
        factor=3,
        directory=f"{Config.MODELS_DIR}resnetturner/",
        project_name="food_classification_resnet",
    )

    # Early Stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Search for the best hyperparameters
    tuner.search(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping],
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best Hyperparameters:\n{best_hps.values}")

    # Build the best model with the best hyperparameters
    resnet_model = tuner.hypermodel.build(best_hps)

    csv_logger = CSVLogger(
        f"{Config.LOGS_DIR}resnet_model_{Config.N_CLASSES}_class.log"
    )
    # Train the best model
    history_resnet_model = resnet_model.fit(
        generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, csv_logger],
    )

    # Save the best model and weights
    resnet_model.save(f"{Config.MODELS_DIR}resnet_model_{Config.N_CLASSES}_class.keras")

    return history_resnet_model
