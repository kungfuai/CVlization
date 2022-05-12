from dataclasses import dataclass
import tensorflow as tf


@dataclass
class ImageAugmentation:
    flip_left_right: bool = False
    random_rotation: float = None
    random_brightness: float = 0.2
    random_contrast: float = 1.3
    random_hue: float = 0.1
    random_saturation: float = 0.3

    image_height: int = 32
    image_width: int = 32
    scale_before_crop: float = 1.25
    random_crop: bool = False


def normalize(image, label):
    """Apply per image standardisation to normalize the image"""
    return tf.image.per_image_standardization(image), label


def augment_obsolete(image, label, augmentation_config: ImageAugmentation):
    """Applies augmentations to the given image. For details on the augmentations please refer to the documentation."""
    # First convert image to floating point representation
    if augmentation_config.flip_left_right:
        image = tf.image.random_flip_left_right(image)

    if (augmentation_config.random_brightness or 0) > 0:
        image = tf.image.random_brightness(image, augmentation_config.random_brightness)

    if (augmentation_config.random_contrast or 0) > 1:
        image = tf.image.random_saturation(
            image,
            1 / augmentation_config.random_contrast,
            augmentation_config.random_contrast,
        )

    if (augmentation_config.random_hue or 0) > 0:
        image = tf.image.random_hue(image, augmentation_config.random_hue)

    if (augmentation_config.random_saturation or 0) > 1:
        s = augmentation_config.random_saturation
        image = tf.image.random_saturation(image, 1 / s, s)

    # Randomly increase the size of the image slightly to then randomly crop a part out of it.
    # This is a way to get random scales + translations
    if augmentation_config.random_crop:
        random_height = tf.random.uniform(
            (),
            minval=augmentation_config.image_height,
            maxval=int(
                augmentation_config.image_height * augmentation_config.scale_before_crop
            ),
            dtype=tf.int32,
        )
        random_width = tf.random.uniform(
            (),
            minval=augmentation_config.image_width,
            maxval=int(
                augmentation_config.image_width * augmentation_config.scale_before_crop
            ),
            dtype=tf.int32,
        )
        image = tf.image.resize(image, (random_height, random_width))
        if len(image.shape) == 3:
            n_channels = image.numpy().shape[-1]
            image = tf.image.random_crop(
                image,
                (
                    augmentation_config.image_height,
                    augmentation_config.image_width,
                    n_channels,
                ),
            )
        else:
            batch_size = image.numpy().shape[0]
            n_channels = image.numpy().shape[-1]
            image = tf.image.random_crop(
                image,
                (
                    batch_size,
                    augmentation_config.image_height,
                    augmentation_config.image_width,
                    n_channels,
                ),
            )

    return image, label
