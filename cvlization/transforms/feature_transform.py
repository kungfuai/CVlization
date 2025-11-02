from dataclasses import dataclass


@dataclass
class FeatureTransform:
    """Abstract class. Wraps a sklearn transformer or other preprocessing step."""

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
