class ConfusionMatrix:
    def __init__(
        self,
        true_positives: int = 0,
        false_negatives: int = 0,
        false_positives: int = 0,
    ) -> None:
        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self._precision = None
        self._recall = None
        self._f1 = None

    def add(
        self, true_positives: int, false_negatives: int, false_positives: int
    ) -> None:
        self.true_positives += true_positives
        self.false_negatives += false_negatives
        self.false_positives += false_positives

    @property
    def precision(self) -> float:
        if not isinstance(self._precision, type(None)):
            return self._precision
        else:
            if self.true_positives + self.false_positives == 0:
                self._precision = 0
            else:
                self._precision = (self.true_positives) / (
                    self.true_positives + self.false_positives
                )
        return self._precision

    @property
    def recall(self) -> float:
        if not isinstance(self._recall, type(None)):
            return self._recall
        else:
            if self.true_positives + self.false_negatives == 0:
                self._recall = 0
            else:
                self._recall = (self.true_positives) / (
                    self.true_positives + self.false_negatives
                )
        return self._recall

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0
        else:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)