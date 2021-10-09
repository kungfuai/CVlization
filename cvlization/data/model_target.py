from dataclasses import dataclass
from typing import Optional, List

from .data_column import DataColumn
from ..losses.loss_type import LossType


@dataclass
class ModelTarget(DataColumn):
    loss: Optional[LossType] = None
    loss_weight: Optional[float] = 1
    metrics: Optional[List] = None
