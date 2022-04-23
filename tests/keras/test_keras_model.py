import numpy as np
import tensorflow as tf
from keras.engine.functional import Functional
from cvlization.keras.model import Model
from cvlization.keras import image_backbone_names, create_image_backbone


def test_custom_keras_model_can_save_and_load(tmpdir):
    x = tf.keras.Input(shape=(10,))
    y = tf.keras.layers.Dense(1)(x)
    m = Model(inputs=x, outputs=y)
    model_path = tmpdir.join("model.h5")
    m.save(str(model_path))
    m2 = tf.keras.models.load_model(str(model_path))
    assert type(m) == Model
    assert type(m2) == Functional
    assert len(m.trainable_variables) == len(m2.trainable_variables)
    for v1, v2 in zip(m.trainable_variables, m2.trainable_variables):
        assert v1.name == v2.name
        assert v1.shape == v2.shape
        assert v1.dtype == v2.dtype
        assert np.all(v1.numpy() == v2.numpy())
    m3 = Model.from_functional_model(m2, n_gradients=2)
    assert type(m3) == Model
    assert m3._n_gradients == 2
    assert len(m3.trainable_variables) == len(m.trainable_variables)
    for v1, v2 in zip(m3.trainable_variables, m.trainable_variables):
        assert v1.name == v2.name
        assert v1.shape == v2.shape
        assert v1.dtype == v2.dtype
        assert np.all(v1.numpy() == v2.numpy())


def test_can_get_image_backbone_names():
    names = image_backbone_names()
    assert "ResNet50" in names
    assert "vit_b32" in names
    assert "convnext_tiny_224" in names


def test_can_create_common_backbones():
    assert create_image_backbone(name="ResNet50", pretrained=False)
    # For VIT, input shape needs to be 3 integers. And the last one needs to be 3.
    assert create_image_backbone(
        name="vit_b32", pretrained=False, input_shape=[1216, 800, 3]
    )
    assert create_image_backbone(name="convnext_tiny_224", pretrained=False)
