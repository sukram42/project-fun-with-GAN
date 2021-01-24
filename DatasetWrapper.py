"""
File to define classes to load Datasets.
"""

from typing import Tuple
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


class DataSet:
    """
    Super Class for Datasets. Implements the main methods to scale the images between -1 and 1,
    and furthermore creates an onehot encoding of the images.

    The data can be accessed via the get_data method.
    """
    NUM_CLASSES: int

    x_data: np.array
    y_data: np.array

    img_x_size: int
    img_y_size: int
    channels: int

    def __init__(self):
        # Unpacking the data
        self._unpack_data(self.get_dataset())
        self._set_image_sizes()
        self._normalize_data()

    def _normalize_data(self):
        self.x_data = self.x_data.reshape((-1, self.img_x_size, self.img_x_size, 1))

        # One Hot Encode Data
        self.y_data = tf.one_hot(self.y_data, self.NUM_CLASSES)

        # Normalize images to be between -1 and 1
        self.x_data = (self.x_data - 127.5) / 127.5

    def _unpack_data(self, dataset):
        """
        Private Method to load the data into the class
        :param data:
        :return:
        """
        (x_train, y_train), (x_test, y_test) = dataset
        self.x_data = np.concatenate([x_train, x_test])
        self.y_data = np.concatenate([y_train, y_test])

    def get_dataset(self) -> Tuple[Tuple[np.array], Tuple[np.array]]:
        """
        Method which loads the dataset. Returns a tuple of following structure:
            (x_train, y_train), (x_test, y_test)
        :return:
        """
        raise NotImplementedError

    def get_data(self, amount=-1):
        """
        Returns the dataset
        If it is true -> (X,Y) is returned
        :return:
        """
        if amount == -1:
            return self.x_data, self.y_data
        return self.x_data[:amount], self.y_data[:amount]

    def show_example(self):
        """
        Method to plot an example
        :return:
        """
        rand_index = np.random.randint(low=0, high=len(self.x_data))
        plt.imshow(self.x_data[rand_index].reshape((self.img_x_size, self.img_x_size)), cmap="gray")

    def _set_image_sizes(self):
        print(len(self.x_data.shape))
        _, self.img_x_size, self.img_y_size = self.x_data.shape


class MNISTDataSet(DataSet):
    """
    Class for the simple MNIST Dataset
    """
    NUM_CLASSES = 10

    def __init__(self, path="mnist.npz"):
        self.path = path
        super().__init__()

    def get_dataset(self):
        return tf.keras.datasets.mnist.load_data(path=self.path)


class FashionMNIST(DataSet):
    """
    Class for the FashionMNIST Dataset
    """
    NUM_CLASSES = 10

    def get_dataset(self):
        return tf.keras.datasets.fashion_mnist.load_data()


class CIFAR10(DataSet):
    NUM_CLASSES = 10

    def get_dataset(self):
        return tf.keras.datasets.cifar10.load_data()


if __name__ == '__main__':
    ds = CIFAR10()
