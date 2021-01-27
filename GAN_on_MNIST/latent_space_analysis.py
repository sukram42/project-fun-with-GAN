import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.animation as anim

import numpy as np
model = tf.saved_model.load("./generator_model_latent_2")


def get_random_seed(shape=(1, 1, 2)):
    """Method to create a random seed"""
    return tf.random.normal(shape=shape)


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ims =[]

for i in range(-30, 32, 2):
    print(i)
    for j in range(-30, 32, 2):
        seed = tf.constant([[[i/10, j/10]]]) #get_random_seed()
        # seed = tf.convert_to_tensor(np.array([i, j]).reshape((1, 1, 2)))
        im1 = ax[0].imshow(model(seed)[0].numpy().reshape((28, 28)), cmap="gray")
        ax[1].set_xlim([-3, 3])
        ax[0].set_title("Generated sample")
        ax[1].set_ylim([-3, 3])
        ax[1].set_title("Latent Representation")
        im2 = ax[1].scatter(seed[0][0][0], seed[0][0][1], color="red")

        ims.append([im1, im2])

ani = anim.ArtistAnimation(fig, ims, interval=50, blit=False)
print("saving")
ani.save("subplots.mp4")
plt.show()

if __name__ == '__main__':
    pass
