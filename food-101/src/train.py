import keras.api.utils as K
from keras.api.models import load_model, Sequential, Model
from keras.api.applications.inception_v3 import InceptionV3
from keras.api.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPool2D,
    ZeroPadding2D,
    GlobalAvgPool2D,
    AvgPool2D,
)
from keras.src.legacy.preprocessing import image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.api.optimizers import SGD
from keras.api.regularizers import L2
import keras
from config import Config


def second_method():
    # Create a sequential model
    model = Sequential()

    # Add convolutional layers with activation and pooling
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3)))
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

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    
    csv_logger = CSVLogger("history_3class.log")

    # Train the model with early stopping
    history = model.fit(
        generator,
        epochs=5,
        validation_data=validation_generator,
        callbacks=[early_stopping, csv_logger],
    )
    
    model.save("model_trained_3class.keras")


def fine_tune_model():
    K.clear_session()

    # Rescales the pixel values to the range [0, 1] (from [0, 255]).
    # Applies a shear transformation to the images.
    # Randomly zooms into the images by 20%.
    # Randomly flips the images horizontally.

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Loads images from the training directory using the train_datagen with augmentations.

    train_generator = train_datagen.flow_from_directory(
        Config.MINI_TRAIN_DATA_DIR,
        target_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode="categorical",
    )

    validation_generator = test_datagen.flow_from_directory(
        Config.MINI_TEST_DATA_DIR,
        target_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode="categorical",
    )

    #  Loads the pre-trained InceptionV3 model with weights trained on ImageNet. The include_top=False argument
    #  excludes the fully connected layers at the top of the model so that you can add custom layers
    inception = InceptionV3(weights="imagenet", include_top=False)
    x = inception.output
    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    predictions = Dense(3, kernel_regularizer=L2(0.005), activation="softmax")(x)

    model = Model(inputs=inception.input, outputs=predictions)
    model.compile(
        optimizer=SGD(learning_rate=0.0001, momentum=0.9),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    checkpointer = ModelCheckpoint(
        filepath="best_model_3class.keras", verbose=1, save_best_only=True
    )
    csv_logger = CSVLogger("history_3class.log")

    history = model.fit(
        train_generator,
        steps_per_epoch=Config.TRAIN_SAMPLE_SIZE // Config.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=Config.TEST_SAMPLE_SIZE // Config.BATCH_SIZE,
        epochs=30,
        verbose=1,
        callbacks=[csv_logger, checkpointer],
    )

    model.save("model_trained_3class.keras")


if __name__ == "__main__":
    second_method()
