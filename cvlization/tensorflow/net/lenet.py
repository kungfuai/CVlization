from tensorflow.keras import models, layers


def LeNet():
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(layers.MaxPool2D(strides=2))
    model.add(
        layers.Conv2D(
            filters=48, kernel_size=(5, 5), padding="valid", activation="relu"
        )
    )
    model.add(layers.MaxPool2D(strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(84, activation="relu"))
    return model
