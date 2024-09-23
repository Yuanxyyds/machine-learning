from keras.api.preprocessing import image
from config import Config
import numpy as np
from keras.src.saving import load_model


def predict(model_path, image_path):
    """
    Loads a trained model and uses it to predict the class of a given image.

    This function:
    1. Loads a pre-trained model from the specified `model_path`.
    2. Preprocesses the input image, resizing it to the dimensions required by the model.
    3. Performs a prediction using the model and returns the predicted class from `Config.FOOD_LIST`.

    Args:
        model_path (str): The file path to the pre-trained model (in Keras format).
        image_path (str): The file path to the image that will be classified.

    Returns:
        str: The predicted food class from `Config.FOOD_LIST`.

    Example:
        result = predict("path/to/model.h5", "path/to/image.jpg")
        print(result)  # Outputs the predicted food class

    Note:
        The image is resized to the dimensions specified in `Config.IMAGE_HEIGHT` and `Config.IMAGE_WIDTH`
        and normalized by dividing pixel values by 255.
    """
    # Load model
    model = load_model(model_path)
    # Load and preprocess the image
    img = image.load_img(
        image_path, target_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    print(Config.FOOD_LIST[predicted_class])
    return Config.FOOD_LIST[predicted_class]
