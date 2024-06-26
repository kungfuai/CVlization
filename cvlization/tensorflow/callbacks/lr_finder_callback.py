import logging
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


LOGGER = logging.getLogger(__name__)


class LRFinderCallback(Callback):
    """Adapted from: https://github.com/WittmannF/LRFinder"""

    def __init__(
        self,
        min_lr,
        max_lr,
        mom=0.9,
        stop_multiplier=None,
        reload_weights=True,
        batches_lr_update=5,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p["epochs"] * p["samples"] // p["batch_size"]
        except:
            n_iterations = p["steps"] * p["epochs"]

        self.learning_rates = np.geomspace(
            self.min_lr, self.max_lr, num=n_iterations // self.batches_lr_update + 1
        )
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        if self.reload_weights:
            self.model.save_weights("tmp.hdf5")

    def on_batch_end(self, batch, logs={}):
        loss = logs.get("loss")

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1] * self.mom + loss * (1 - self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if (
            self.iteration % self.batches_lr_update == 0
        ):  # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights("tmp.hdf5")

            lr = self.learning_rates[self.iteration // self.batches_lr_update]
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)

        if self.best_loss is None:
            LOGGER.error(f"Best loss is None")

        if loss > self.best_loss * self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights("tmp.hdf5")

        # plt.figure(figsize=(12, 6))
        # plt.plot(self.learning_rates[: len(self.losses)], self.losses)
        # plt.xlabel("Learning Rate")
        # plt.ylabel("Loss")
        # plt.xscale("log")
        # plt.show()

    def best_lr(self):
        return self.learning_rates[np.argmin(self.losses)]


if __name__ == "__main__":
    lrfinder = LRFinderCallback(0.00001, 0.1)
    lrfinder.learning_rates = [0.01, 0.1, 0.0005]
    lrfinder.losses = [1, 10, -1]
    print(lrfinder.best_lr())
