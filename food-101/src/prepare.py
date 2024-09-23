"""
Updated Sep 23, 2024 by Steven Liu

prepare.py contains initialization code to warm up your device before training. 
It detects whether the CPU and GPU are properly configured and utilized. If the GPU 
is not correctly configured, please ensure that your GPU drivers are installed and functioning correctly.

Note: The training process will be significantly slower if the GPU is not properly installed.
"""

import tensorflow as tf
import time


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
        
