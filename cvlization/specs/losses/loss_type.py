import enum


class LossType(enum.Enum):
    BINARY_CROSSENTROPY = "BinaryCrossentropy"
    CATEGORICAL_CROSSENTROPY = "CategoricalCrossentropy"
    SPARSE_CATEGORICAL_CROSSENTROPY = "SparseCategoricalCrossentropy"
    MSE = "MSE"
    MAE = "MAE"
    MAPE = "mape"
    LOG = "log"
