from dataclasses import dataclass


@dataclass
class FeatureTransform:
    """Wraps a sklearn transformer, imgaug step, etc."""

    to_cache: bool

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass

    def save(self):
        pass

    def load(self, fn):
        pass
