import tensorflow as tf

model = tf.saved_model.load("./generator_model")


def get_random_seed(shape=(1024, 100)):
    """Method to create a random seed"""
    return tf.random.normal(shape=shape)


if __name__ == '__main__':
    pass
