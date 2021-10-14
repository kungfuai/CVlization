from typing import List, Optional
import enum
from dataclasses import dataclass, field as dataclass_field
import logging

from ..transforms.feature_transform import FeatureTransform


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class DataColumnType(enum.Enum):
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
    # TODO: should transforms track the shape
    # transforms include fixed transforms and random augmentation
    # TODO: should allow user to easily
    #   implement data fetching, and transforms
    transforms: List[FeatureTransform] = None

    @property
    def shape(self):
        # infer from transforms
        if len(self.transforms or []) == 0:
            return self.raw_shape
