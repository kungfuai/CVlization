import numpy as np
import tensorflow as tf

from ..data.ml_dataset import MLDataset


def _to_tensor_spec(array: np.ndarray) -> tf.TensorSpec:
    shape = (None,) + tuple(array.shape[1:])
    dtype = tf.as_dtype(array.dtype)
    return tf.TensorSpec(shape=shape, dtype=dtype)


def from_ml_dataset(
    ml_dataset: MLDataset, batch_size: int = None, repeat: bool = False
) -> tf.data.Dataset:
    """Convert an MLDataset to a tf.data.Dataset compatible with Keras 3.

    Each element in the dataset is a tuple of (inputs, targets, sample_weights),
    where inputs and targets are tuples of tensors in the same order as the
    model inputs/targets, and sample_weights is a 1-D float32 tensor.
    """

    batch_size = batch_size if batch_size is not None else ml_dataset.batch_size
    num_examples = len(ml_dataset)
    if num_examples == 0:
        raise ValueError("MLDataset is empty.")

    first_end = min(batch_size, num_examples)
    first_inputs, first_targets, first_sample_weight = ml_dataset.get_batch_from_range(
        0, first_end
    )
    first_inputs = [np.asarray(arr) for arr in first_inputs]
    first_targets = [np.asarray(arr) for arr in first_targets]

    if first_sample_weight is None:
        first_sample_weight = np.ones((first_inputs[0].shape[0],), dtype=np.float32)
    else:
        first_sample_weight = np.asarray(first_sample_weight, dtype=np.float32)
        if first_sample_weight.ndim > 1:
            first_sample_weight = first_sample_weight.reshape(first_sample_weight.shape[0])

    input_signature = tuple(_to_tensor_spec(arr) for arr in first_inputs)
    target_signature = tuple(_to_tensor_spec(arr) for arr in first_targets)
    sample_signature = tf.TensorSpec(shape=(None,), dtype=tf.float32)

    def generator():
        start = 0
        cached_first = (
            tuple(first_inputs),
            tuple(first_targets),
            first_sample_weight,
        )
        use_cached_first = True
        while start < num_examples:
            if use_cached_first:
                inputs_batch, targets_batch, sample_weight_batch = cached_first
                use_cached_first = False
            else:
                end_idx = min(start + batch_size, num_examples)
                inputs_batch, targets_batch, sample_weight_batch = ml_dataset.get_batch_from_range(
                    start, end_idx
                )
                inputs_batch = [np.asarray(arr) for arr in inputs_batch]
                targets_batch = [np.asarray(arr) for arr in targets_batch]
                if sample_weight_batch is None:
                    sample_weight_batch = np.ones(
                        (inputs_batch[0].shape[0],), dtype=np.float32
                    )
                else:
                    sample_weight_batch = np.asarray(sample_weight_batch, dtype=np.float32)
                    if sample_weight_batch.ndim > 1:
                        sample_weight_batch = sample_weight_batch.reshape(
                            sample_weight_batch.shape[0]
                        )
                inputs_batch = tuple(inputs_batch)
                targets_batch = tuple(targets_batch)
            yield inputs_batch, targets_batch, sample_weight_batch
            start += batch_size

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(input_signature, target_signature, sample_signature),
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)
