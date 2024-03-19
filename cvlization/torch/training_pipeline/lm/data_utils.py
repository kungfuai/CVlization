"""
Utilities for process latents and token_ids from videos.
"""
import numpy as np

class FlatTokenIds:

    def __init__(self, token_ids: np.ndarray, vocab_size: int=None, start_token_id: int = 0, train_split: float=0.8):
        """
        Prepare training and validation datasets from 1D or 2D token ids.

        Args:
            token_ids: numpy array of shape (n, m) or (n,)
            vocab_size: size of the vocabulary
            start_token_id: token id for the start token
            train_split: proportion of the dataset to use for training
        """
        assert isinstance(token_ids, np.ndarray), "latents must be a numpy array"
        
        if token_ids.ndim == 2:
            n = token_ids.shape[0]
            n_train = int(n * train_split)
            self.train_token_ids = token_ids[:n_train]
            self.val_token_ids = token_ids[n_train:]
            self.vocab_size = vocab_size
            self.start_token_id = start_token_id
        elif token_ids.ndim == 1:
            n = token_ids.shape[0]
            n_train = int(n * train_split)
            self.train_token_ids = token_ids[:n_train]
            self.val_token_ids = token_ids[n_train:]
        else:
            raise ValueError("token ids must be 1D or 2D")

    
    def _insert_start_token(self, token_ids: np.ndarray, start_token_id: int):
        """
        Insert start token at the beginning of each sequence.
        """
        if start_token_id is None:
            return token_ids
        start_token = np.ones((token_ids.shape[0], 1), dtype=token_ids.dtype) * start_token_id
        return np.concatenate([start_token, token_ids], axis=1)

    def training_dataset(self):
        """
        Return training dataset as one big sequence of token ids.
        """
        token_ids = self._insert_start_token(self.train_token_ids, self.start_token_id)
        return token_ids.ravel()
    
    def validation_dataset(self):
        """
        Return validation dataset as one big sequence of token ids.
        """
        token_ids = self._insert_start_token(self.val_token_ids, self.start_token_id)
        return token_ids.ravel()



