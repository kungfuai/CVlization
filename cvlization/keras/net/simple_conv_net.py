from tensorflow import keras
from tensorflow.keras import layers


def SimpleConvNet(input_shape):
    """A convnet from the Keras MNIST tutorial."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
        ]
    )
    return model
