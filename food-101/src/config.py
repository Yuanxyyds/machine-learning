import tensorflow as tf
import time


class Config:
    BATCH_SIZE = 16
    N_CLASSES = 3
    IMAGE_WIDTH = 299
    IMAGE_HEIGHT = 299
    DATA_DIR = "food-101/data/images/"
    TRAIN_DATA_METADATA = "food-101/data/meta/train.txt"
    TEST_DATA_METADATA = "food-101/data/meta/test.txt"
    MINI_TRAIN_DATA_DIR = "food-101/processed_data/mini_train"
    MINI_TEST_DATA_DIR = "food-101/processed_data/mini_test"
    TRAIN_DATA_DIR = "food-101/processed_data/train"
    TEST_DATA_DIR = "food-101/processed_data/test"
    FOOD_LIST = ["apple_pie", "pizza", "chicken_wings"]
    TRAIN_SAMPLE_SIZE = -1
    TEST_SAMPLE_SIZE = -1


def check_gpu_init():
    """
    Check if GPU is properly initialized
    """
    print(tf.__version__)
    print(tf.test.gpu_device_name())


def hardware_exam():
    """
    Test Computing time for GPU and CPU on Matrix multiplication
    """
    with tf.device("/GPU:0"):
        start_time = time.time()
        A = tf.random.normal([10000, 10000])
        B = tf.random.normal([10000, 10000])
        _ = tf.matmul(A, B)
        print("GPU Operation completed in:", time.time() - start_time, "seconds")

    with tf.device("/CPU:0"):
        start_time = time.time()
        A = tf.random.normal([10000, 10000])
        B = tf.random.normal([10000, 10000])
        _ = tf.matmul(A, B)
        print("CPU operation completed in:", time.time() - start_time, "seconds")


if __name__ == "__main__":
    check_gpu_init()
    hardware_exam()
