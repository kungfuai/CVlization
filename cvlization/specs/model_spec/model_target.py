from dataclasses import dataclass
from typing import Optional, List, Union

from ..data_column import DataColumn, DataColumnType
from ..losses.loss_type import LossType

"""
TODO: allow a model_target to specify an aggregator, and inputs to the aggregator
TODO: consider having multiple ModelPath. Each model path specifies
        a different of model inputs and model targets.
TODO: consider having model input groups.

ModelInput:
    key: key1
    input_groups: [default, group1, group2]
ModelInput:
    key: key1
    input_groups: [group1]
ModelTarget:
    input_group: group1
    aggregator: str
ModelTarget:
    input_group: group2
    target_groups: [default, g1]
EnsembleModelTarget:
    target_group_to_ensemble: g1

TODO: allow implementing a new aggregator and registering it.
"""


@dataclass
class ModelTarget(DataColumn):
    loss: Optional[LossType] = None
    loss_weight: Optional[float] = 1
    metrics: Optional[List] = None

    # For imbalanced label distribution.
    negative_class_weight: Optional[float] = 1

    # For monotone constraints on a lattice of output values.
    monotone_constraint_key: Optional[str] = "default_monotone_constraint"
    monotone_constraint_rank: Optional[int] = None  # From small to large.

    # For loss functions.
    prefer_logits: Optional[bool] = True
    learning: Optional[
        bool
    ] = True  # If False, this target is used for unlearning (adversarial loss).

    # For allowing additional tweaks on the neural network architecture.
    # Allow individual feedback from this target to the group of inputs.
    target_groups: Optional[List[str]] = None
    input_group: Optional[str] = None

    def __post_init__(self):
        if self.column_type == DataColumnType.CATEGORICAL:
            assert (
                self.n_categories > 1
            ), "Categorical target must have more than one category"
        self.target_groups = self.target_groups or []

    @classmethod
    def DEFAULT_TARGET_GROUP(cls):
        return "default"


@dataclass
class EnsembleModelTarget(ModelTarget):
    """A ModelTargetForEnsemble aggregates outputs from multiple sub-models.

    target_keys_to_ensemble: a list of target keys to be aggregated. The output
        tensors of these targets will be used.
    """

    target_group_to_ensemble: Optional[str] = None
    aggregation_method: Optional[str] = "avg"  # "avg" or "max"
