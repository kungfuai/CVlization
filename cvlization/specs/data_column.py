from typing import List, Optional, Union
import enum
from dataclasses import dataclass, field as dataclass_field
import logging

from ..transforms.feature_transform import FeatureTransform


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class DataColumnType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    # Structured outputs:
    BOUNDING_BOXES = "bbox"
    LABAL_MAP = "label_map"
    KEYPOINTS = "keypoints"


@dataclass
class DataColumn:
    """
    Can cross check with the Dataset to make sure it generates the connect
    number, type and shape of targets.

    Similar to tensorflow feature column.

    Want to use this for both tf and torch.
    """

    key: str
    raw_shape: Optional[List] = None
    column_type: Optional[DataColumnType] = DataColumnType.NUMERICAL

    # For a categorical column, this is the number of levels.
    n_categories: Optional[int] = None

    # Indicate whether the numpy array is variable sized sequence, e.g. bounding boxes.
    # Usually it is enough to set sequence to True when you have such target variables.
    # But if you have multilple groups of such sequences, each one with a different length,
    # you can set `sequence` to a unique str value for each group of sequences.
    # For a goup of sequences, the size of their "sequence" axis are expected to match.
    # For example, two sequence targets bbox_labels (of shape [n, 1]) and bboxes (of shape [n, 4])
    # are expected to have the same sequence length n.
    sequence: Optional[Union[bool, str]] = False

    # TODO: transforms should track the shape
    # transforms include fixed transforms and random augmentation
    # TODO: should allow user to easily
    #   implement data fetching, and transforms
    transforms: List[FeatureTransform] = None

    def __post_init__(self):
        if self.raw_shape is None:
            self._fill_in_missing_raw_shape()

    def _fill_in_missing_raw_shape(self):
        if self.column_type == DataColumnType.BOOLEAN:
            self.raw_shape = [1]

    @property
    def shape(self):
        # infer from transforms
        if len(self.transforms or []) == 0:
            return self.raw_shape
