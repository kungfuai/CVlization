class ConfusionMatrix:
    def __init__(self, true_positives: int=0, false_negatives: int=0, false_positives: int=0) -> None:
        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.false_positives = false_positives

    def add(self, true_positives: int, false_negatives: int, false_positives: int) -> None:
        self.true_positives += true_positives
        self.false_negatives += false_negatives
        self.false_positives += false_positives

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0
        return (self.true_positives) / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0
        return (self.true_positives) / (self.true_positives + self.false_negatives)
