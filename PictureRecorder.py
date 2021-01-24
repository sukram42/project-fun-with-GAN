from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PictureRecorder:
    """
    Class to record images to see improvement of the Generator
    """
    _imgs: List
    _channel: int
    _fig: plt.Figure

    def __init__(self, channel=1):
        self._imgs = []
        self._channel = channel
        self._fig = plt.figure()

    def add(self, image):
        """Add an image to the recorder"""
        _img = plt.imshow(image, cmap="gray")
        self._imgs.append([_img,_img])

    def save_movie(self, file="image.gif"):
        """
        Method to save the record
        :param file:
        :return:
        """
        print(f"Got {len(self._imgs)} frames")
        print(self._imgs)
        ani = animation.ArtistAnimation(self._fig, self._imgs,
                                        interval=50, repeat_delay=1000)
        ani.save(file)


if __name__ == '__main__':
    recorder = PictureRecorder()
    for i in range(3):
        x = np.random.randint(0, 254, size=(28, 28))
        recorder.add(x)
    recorder.save_movie("testmov.mp4")
