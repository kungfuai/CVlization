import math
import sys
from tensorflow import keras

from ..data.ml_dataset import MLDataset


def from_ml_dataset(
    ml_dataset: MLDataset, batch_size: int = None
) -> keras.utils.Sequence:

    batch_size = batch_size if batch_size is not None else ml_dataset.batch_size

    class GeneratedSequence(keras.utils.Sequence):
        def __init__(self):
            self.ml_dataset = ml_dataset
            self.n_examples = len(ml_dataset)
            self.batch_size = batch_size
            self.n_batches = int(math.ceil(self.n_examples / float(batch_size)))

        def __getitem__(self, i):
            """
            Returns:
                The i-th batch. The batch is a tuple: (inputs_batch, targets_batch).
                inputs_batch and targets_batch each should be a list of numpy arrays.
                inputs_batch has len(model_inputs) input arrays.
                targets_batch has len(model_targets) target arrays.
            """
            begin_idx = i * batch_size
            end_idx = min(i * batch_size + batch_size, self.n_examples)
            try:
                (
                    inputs_batch,
                    targets_batch,
                    sample_weight_batch,
                ) = ml_dataset.get_batch_from_range(begin_idx, end_idx)
            except KeyboardInterrupt:
                sys.exit(1)
            except Exception:
                raise
            return inputs_batch, targets_batch, sample_weight_batch

        def __len__(self):
            return self.n_batches

    return GeneratedSequence()
