import enum


class MLFramework(str, enum.Enum):
    # Using str together as Enum for easy json serialization.
    # See: https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    MXNET = "mxnet"
