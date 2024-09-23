"""
This script provides utility functions to manage and visualize the Food-101 dataset for a food classification model.

Functions:
    - visualize_dataset(): Visualizes a subset of the Food-101 dataset by displaying one image per class, organized in a grid.
    - split_train_and_test_data(): Splits the raw data into separate training and testing directories based on predefined metadata files.
    - split_mini_train_and_test_data(): Creates a smaller subset of the dataset by copying a specified list of food classes into new directories.

Dependencies:
    - os: For directory and file path manipulation.
    - matplotlib.pyplot: For visualizing images from the dataset.
    - numpy: For random selection of images.
    - shutil: For copying files and directories.
    - collections.defaultdict: For organizing and grouping images by class.
    - config.Config: Imports dataset configuration and paths.

Usage:
    This script is intended to help prepare and manage the dataset for training a food classification model.
    It can be used to visualize the dataset, split the raw data into training and testing sets, 
    and create smaller, more manageable subsets of the data for testing and experimentation.

Ensure the configuration in `config.Config` is properly set before running these functions.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy, copytree, rmtree
from collections import defaultdict
from config import Config


def visualize_dataset():
    """
    Visualizes a subset of the dataset by displaying one image per class from the available food categories.

    This function generates a grid of images, with each cell displaying a randomly selected image 
    from one of the food categories present in the dataset. The number of rows and columns in the grid 
    is determined by the `rows` and `cols` variables.

    The output is saved as an image in the folder specified by `Config.PLOTS_DIR`.

    Behavior:
        - If the directory specified by `Config.PLOTS_DIR` does not exist, it is created.
        - The images are displayed without axes ticks.
        - The result is saved as a PNG file in `Config.PLOTS_DIR`.

    Output:
        - A saved image showing the visualization of food categories.
    """
    # Make the plots folder if is not included
    if not os.path.exists(Config.PLOTS_DIR):
        os.makedirs(Config.PLOTS_DIR)
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
    plt.savefig(f"{Config.PLOTS_DIR}visualize_dataset.png")


def split_train_and_test_data():
    """
    Splits the raw data into training and testing sets.

    This function uses metadata files (`train.txt` and `test.txt`) to identify which images belong to 
    the training and testing sets, respectively. The images are then copied from the source directory 
    (`Config.DATA_DIR`) to their respective training and testing directories.

    Behavior:
        - Training images are copied to `Config.TRAIN_DATA_DIR`.
        - Testing images are copied to `Config.TEST_DATA_DIR`.
        - If the destination directories do not exist, they are created.

    Output:
        - The training and testing datasets are copied into their respective directories.
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
    Creates smaller subsets of the training and testing data, using a limited number of food categories.

    This function copies only the categories listed in `Config.FOOD_LIST` from the full training 
    and testing sets to separate directories for faster prototyping and testing.

    Behavior:
        - The function removes any existing data in `Config.MINI_TRAIN_DATA_DIR` and `Config.MINI_TEST_DATA_DIR`.
        - Copies images for the food categories specified in `Config.FOOD_LIST` into the mini train/test directories.
        - Counts and logs the number of images copied into each directory.

    Output:
        - The mini training and testing datasets are copied into their respective directories.
        - Logs the total number of images copied.
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