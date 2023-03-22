"""
Adapted from https://github.com/huggingface/transformers/blob/f48d3314e42bf54accc9dd8fd8dc1bf4197b34c6/src/transformers/utils/import_utils.py#L63
"""
import importlib.util

_tensorflow_available = importlib.util.find_spec("tensorflow") is not None
_tensorflow_tpu_available = _tensorflow_available and importlib.util.find_spec("tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver") is not None
_tf2_available = _tensorflow_available and importlib.util.find_spec("tensorflow.compat.v2") is not None

_torch_available = importlib.util.find_spec("torch") is not None
_torch_fx_available = importlib.util.find_spec("torch.fx") is not None
_torch_geometric_available = importlib.util.find_spec("torch_geometric") is not None
_torchvision_available = importlib.util.find_spec("torchvision") is not None
_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
_torchtext_available = importlib.util.find_spec("torchtext") is not None

_skimage_available = importlib.util.find_spec("skimage") is not None

def is_tf_available():
    return _tensorflow_available

def is_torch_available():
    return _torch_available

def is_skimage_available():
    return _skimage_available
