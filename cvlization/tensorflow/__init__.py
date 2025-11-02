"""TensorFlow-specific implementations."""

# Avoid eager imports at package level to prevent conflicts with `import tensorflow`
# Users should use explicit imports when possible; however, common helpers are
# re-exported here for backwards compatibility with older tests and examples.

from .encoder.keras_image_backbone import (
    image_backbone_names,
    create_image_backbone,
)

__all__ = [
    "image_backbone_names",
    "create_image_backbone",
]
