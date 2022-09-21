from typing import Tuple, Any
from torch import nn


class TorchModuleSurgery:
    """An abstract class for torch modules.
    """
    def run(self, model: nn.Module, **kwargs) -> Tuple[nn.Module, Any]:
        raise NotImplementedError
