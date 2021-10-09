import enum


class LossType(enum.Enum):
    BINARY_CROSSENTROPY = "BinaryCrossentropy"
    CATEGORICAL_CROSSENTROPY = "CategoricalCrossentropy"
    MSE = "MSE"
    MAE = "MAE"
    MAPE = "mape"
    LOG = "log"
