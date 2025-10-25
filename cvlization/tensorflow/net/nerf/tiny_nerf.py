# Adapted from https://github.com/bmild/nerf
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.engine import data_adapter


LOGGER = logging.getLogger(__name__)
N_samples = 64
N_iters = 2000
psnrs = []
iternums = []
i_plot = 50


class TinyNerfModel(tf.keras.Model):
    def __init__(self, D=4, W=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_eagerly = True
        self.coordinate_model = init_model(D, W)

    def call(self, inputs):
        poses = inputs[0]
        focals = inputs[1]
        Hs = inputs[2]
        Ws = inputs[3]
        rgbs = []
        for k in range(len(poses)):
            pose = poses[k]
            focal = focals[k]
            H = float(Hs[k])
            W = float(Ws[k])
            rays_o, rays_d = get_rays(H, W, focal, pose)
            rgb, depth, acc = render_rays(
                self.coordinate_model,
                rays_o,
                rays_d,
                near=2.0,
                far=6.0,
                N_samples=N_samples,
                rand=True,
            )
            rgbs.append(rgb)
        return [tf.stack(rgbs, axis=0)]

    def train_step(self, data):
        assert tf.executing_eagerly(), "Eager execution expected."
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = data_adapter.expand_1d(data)
        inputs, targets, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # print("inputs", inputs)

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            images = targets[0]
            pred_images = y_pred[0]
            # loss is expected to be MSE
            loss = self.compute_loss(None, images, pred_images)
            # loss = self.compiled_loss(
            #     images, pred_images, regularization_losses=self.losses
            # )
            gradients = tape.gradient(loss, self.coordinate_model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.coordinate_model.trainable_variables)
            )

        return_metrics = self._evaluate_metrics(y=targets, y_pred=y_pred, x=inputs)
        # Return a dict mapping metric names to current value
        return return_metrics

    def train_step_old(self, data):
        assert tf.executing_eagerly(), "Eager execution expected."
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = data_adapter.expand_1d(data)
        inputs, targets, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # print("inputs", inputs)
        images = targets[0]
        H = images.shape[1]
        W = images.shape[2]
        poses = inputs[0]
        focals = inputs[1]

        for k in range(len(images)):
            target_image = images[k]
            pose = poses[k]
            focal = focals[k]
            with tf.GradientTape() as tape:
                rays_o, rays_d = get_rays(H, W, focal, pose)
                rgb, depth, acc = render_rays(
                    self.coordinate_model,
                    rays_o,
                    rays_d,
                    near=2.0,
                    far=6.0,
                    N_samples=N_samples,
                    rand=True,
                )
                loss = tf.reduce_mean(tf.square(rgb - target_image))
            gradients = tape.gradient(loss, self.coordinate_model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.coordinate_model.trainable_variables)
            )

        # TODO: return metrics
        return None

    def _evaluate_metrics(self, y, y_pred, x):
        if isinstance(y_pred, EagerTensor):
            y_pred = [y_pred]
        return_metrics = {}
        LOGGER.debug(f"{len(self.metrics)} metrics")
        LOGGER.debug(self.metrics)
        LOGGER.debug("compiled:", self.compiled_metrics)
        # self.metrics[0] is the mean loss.
        # Check if user metrics exist (TF 2.x compatibility fix)
        if len(self.metrics) < 2:
            LOGGER.debug("No user metrics defined, returning empty metrics")
            return return_metrics

        compiled_metrics = self.metrics[1]
        for target_idx, metrics_for_this_target in enumerate(
            compiled_metrics._user_metrics
        ):
            if not isinstance(metrics_for_this_target, list):
                metrics_for_this_target = [metrics_for_this_target]
            for metric in metrics_for_this_target:
                LOGGER.debug(f"To update metric: {metric} for target {target_idx}")
                LOGGER.debug(f"y: {len(y)}, {type(y)}, {y[0].shape}")
                LOGGER.debug(
                    f"y_pred: {len(y_pred)}, {type(y_pred)}, {y_pred[0].shape}"
                )
                if hasattr(metric, "update_state_with_inputs_and_outputs"):
                    metric.update_state_with_inputs_and_outputs(
                        y[target_idx], y_pred[target_idx], train_example=x
                    )
                else:
                    try:
                        metric.update_state(y[target_idx], y_pred[target_idx])
                    except Exception as e:
                        LOGGER.error(f"Failed to update metric {metric}. Target idx: {target_idx}")
                        raise e

        for metric in compiled_metrics.metrics:
            try:
                result = metric.result()
            except:
                LOGGER.error(f"Failed to get result for metric {metric}")
                # raise
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2.0**i * x))
    return tf.concat(rets, -1)


L_embed = 6
embed_fn = posenc
# L_embed = 0
# embed_fn = tf.identity


def init_model(D=8, W=256):
    print(f"Initializing model. L_embed={L_embed}, D={D}, W={W}")
    relu = tf.keras.layers.ReLU()
    dense = lambda W=W, act=relu: tf.keras.layers.Dense(W, activation=act)

    inputs = tf.keras.Input(shape=(3 + 3 * 2 * L_embed,))
    outputs = inputs
    for i in range(D):
        outputs = dense()(outputs)
        if i % 4 == 0 and i > 0:
            outputs = tf.concat([outputs, inputs], -1)
        print("outputs:", outputs)
    final_dense = tf.keras.layers.Dense(4)
    outputs = final_dense(outputs)
    print("outputs:", outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(
        tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing="xy"
    )
    dirs = tf.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -tf.ones_like(i)], -1
    )
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    def batchify(fn, chunk=1024 * 32):
        return lambda inputs: tf.concat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    # Compute 3D query points
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
        z_vals += (
            tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples])
            * (far - near)
            / N_samples
        )
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = tf.reshape(pts, [-1, 3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[..., 3])
    rgb = tf.math.sigmoid(raw[..., :3])

    # Do volume rendering
    dists = tf.concat(
        [
            z_vals[..., 1:] - z_vals[..., :-1],
            tf.broadcast_to([1e10], z_vals[..., :1].shape),
        ],
        -1,
    )
    alpha = 1.0 - tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1.0 - alpha + 1e-10, -1, exclusive=True)

    rgb_map = tf.reduce_sum(weights[..., None] * rgb, -2)
    depth_map = tf.reduce_sum(weights * z_vals, -1)
    acc_map = tf.reduce_sum(weights, -1)

    return rgb_map, depth_map, acc_map
