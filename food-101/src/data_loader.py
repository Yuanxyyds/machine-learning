import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy, copytree, rmtree
from collections import defaultdict
from config import Config


def visualize_dataset():
    """
    Visualize the data, showing one image per class from 101 classes
    """
    rows = 17
    cols = 6
    _, ax = plt.subplots(rows, cols, figsize=(25, 25))
    foods_sorted = sorted(os.listdir(Config.DATA_DIR))
    processed_food = 0
    for i in range(rows):
        for j in range(cols):
            try:
                food_selected = foods_sorted[processed_food]
                processed_food += 1
            except:
                break
            if food_selected == ".DS_Store":
                continue
            food_selected_images = os.listdir(
                os.path.join(Config.DATA_DIR, food_selected)
            )
            food_selected_random = np.random.choice(food_selected_images)
            img = plt.imread(
                os.path.join(Config.DATA_DIR, food_selected, food_selected_random)
            )
            ax[i][j].imshow(img)
            ax[i][j].set_title(food_selected, pad=10)

    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig("food-101/plots/visualize_dataset.png")


def split_train_and_test_data():
    """
    Split the train and test data from the raw data folder
    """

    def extract_from_raw_data(filepath, src, dest):
        image_dict = defaultdict(list)
        with open(filepath, "r") as txt:
            paths = [read.strip() for read in txt.readlines()]
            for p in paths:
                entry = p.split("/")
                image_dict[entry[0]].append(entry[1] + ".jpg")

        for food in image_dict.keys():
            print("\nCopying images into ", food, "......")
            if not os.path.exists(os.path.join(dest, food)):
                os.makedirs(os.path.join(dest, food))
            for image_type in image_dict[food]:
                copy(
                    os.path.join(src, food, image_type),
                    os.path.join(dest, food, image_type),
                )
        print("Copying Done!")

    print("Creating train data...")
    extract_from_raw_data(
        Config.TRAIN_DATA_METADATA,
        Config.DATA_DIR,
        Config.TRAIN_DATA_DIR,
    )
    print("Creating test data...")
    extract_from_raw_data(
        Config.TEST_DATA_METADATA,
        Config.DATA_DIR,
        Config.TEST_DATA_DIR,
    )
    print("Completed...")


def split_mini_train_and_test_data():
    """
    Select less classes of data from the splitted data
    """

    def extract_classes(food_list, src, dest) -> int:
        count = 0
        if os.path.exists(dest):
            rmtree(dest)
        os.makedirs(dest)
        for food_item in food_list:
            print("Copying images into", food_item)
            copytree(os.path.join(src, food_item), os.path.join(dest, food_item))
            count += len(
                [
                    name
                    for name in os.listdir(os.path.join(dest, food_item))
                    if os.path.join(dest, food_item, name)
                ]
            )
        return count

    print("Creating train data folder with new classes")
    Config.TRAIN_SAMPLE_SIZE = extract_classes(
        Config.FOOD_LIST, Config.TRAIN_DATA_DIR, Config.MINI_TRAIN_DATA_DIR
    )
    print("Creating train data folder with new classes")
    Config.TEST_SAMPLE_SIZE = extract_classes(
        Config.FOOD_LIST, Config.TEST_DATA_DIR, Config.MINI_TEST_DATA_DIR
    )
    print("Completed with Training Dataset Size: ", Config.TRAIN_SAMPLE_SIZE)
    print("Completed with Testing Dataset Size: ", Config.TEST_SAMPLE_SIZE)


if __name__ == "__main__":
    split_mini_train_and_test_data()
