from keras.api.preprocessing import image
from config import Config
import numpy as np
from keras.src.saving import load_model


loaded_model = load_model("model_trained_3class.keras")


def predict_and_display(image_path):
    # Load and preprocess the image
    img = image.load_img(
        image_path, target_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)

    print(Config.FOOD_LIST[predicted_class])


if __name__ == "__main__":
    predict_and_display("apple-pie-www.thereciperebel.com-1200-17-of-53.jpg")
