from tensorflow import keras
from tensorflow.keras import layers


def SimpleConvNet(input_shape, pooling: str = "avg", **kwargs):
    """A convnet from the Keras MNIST tutorial."""
    if pooling not in ["avg", "max", "flatten", None]:
        raise ValueError("pooling must be either 'avg', 'max' or None")
    if pooling == "avg":
        pooling_layer = layers.GlobalAveragePooling2D()
    elif pooling == "max":
        pooling_layer = layers.GlobalMaxPooling2D()
    elif pooling == "flatten":
        pooling_layer = layers.Flatten()
    else:
        pooling_layer = None

    model = keras.Sequential(
        [
            l
            for l in [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.LayerNormalization(),
                layers.Conv2D(64, kernel_size=(3, 3), strides=2, activation="relu"),
                layers.LayerNormalization(),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                layers.LayerNormalization(),
                layers.Conv2D(512, kernel_size=(3, 3), activation="relu"),
                layers.LayerNormalization(),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                layers.LayerNormalization(),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.LayerNormalization(),
                pooling_layer,
            ]
            if l is not None
        ]
    )
    return model
