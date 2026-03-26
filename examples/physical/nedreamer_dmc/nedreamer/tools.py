import datetime
import collections
import os
import json
import pathlib
import time
import random
import shutil

import numpy as np

import torch
from torch import nn
import wandb
import boto3
from dotenv import load_dotenv


def to_np(x):
    return x.detach().cpu().numpy()

def to_f32(x):
    return x.to(dtype=torch.float32)

def to_i32(x):
    return x.to(dtype=torch.int32)

class CudaBenchmark:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}
        # wandb will be initialized in log_hydra_config with the full config

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def histogram(self, name, value):
        self._histograms[name] = np.array(value)

    def write(self, step, fps=False):
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps/fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")

        # Prepare wandb log dict
        log_dict = {"step": step}
        for name, value in scalars:
            if "/" not in name:
                log_dict["scalars/" + name] = value
            else:
                log_dict[name] = value
        # NOTE: Image/video/histogram uploads disabled
        # for name, value in self._images.items():
        #     # wandb expects (H, W, C) for images
        #     log_dict[name] = wandb.Image(value)
        # for name, value in self._videos.items():
        #     name = name if isinstance(name, str) else name.decode("utf-8")
        #     if np.issubdtype(value.dtype, np.floating):
        #         value = np.clip(255 * value, 0, 255).astype(np.uint8)
        #     # value shape: (B, T, H, W, C) - wandb.Video expects (T, C, H, W) or (T, H, W, C)
        #     B, T, H, W, C = value.shape
        #     # Concatenate batch dimension horizontally for visualization
        #     value = value.transpose(1, 0, 2, 3, 4).reshape((T, B * H, W, C))
        #     log_dict[name] = wandb.Video(value, fps=16, format="mp4")
        # for name, value in self._histograms.items():
        #     log_dict[name] = wandb.Histogram(value)

        wandb.log(log_dict, step=step)
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        wandb.log({"scalars/" + name: value}, step=step)

    def offline_video(self, name, value, step):
        # NOTE: Video upload disabled
        pass
        # if np.issubdtype(value.dtype, np.floating):
        #     value = np.clip(255 * value, 0, 255).astype(np.uint8)
        # B, T, H, W, C = value.shape
        # value = value.transpose(1, 0, 2, 3, 4).reshape((T, B * H, W, C))
        # wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)

    def log_hydra_config(self, config):
        """
        Initialize wandb and log a Hydra/OmegaConf config.

        Parameters
        ----------
        config : Any
            Hydra config (OmegaConf.DictConfig, etc.)
        """
        container = None
        try:
            from omegaconf import OmegaConf
            container = OmegaConf.to_container(config, resolve=True)
        except Exception:
            container = dict(config) if hasattr(config, '__iter__') else {}

        # Extract run name from logdir or use default
        run_name = self._logdir.name if self._logdir else None

        # Initialize wandb
        wandb.init(
            name=run_name,
            config=container,
            dir=str(self._logdir),
            reinit=True,
        )

def convert(value, precision=32):
    if isinstance(value, dict):
        value = {key: convert(val) for key, val in value.items()}
        return value
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)

def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        use_amp=False,
    ):
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._opt = torch.optim.Adam(self._parameters, lr=lr, eps=eps)
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params):
        assert len(loss.shape) == 0, loss.shape
        # Call this before backward to prevent unintended gradients
        self._opt.zero_grad(set_to_none=True)
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        self._scaler.step(self._opt)
        self._scaler.update()
        metrics = {f"{self._name}_grad_norm": to_np(norm)}
        return metrics


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until

def tensorstats(tensor, prefix=None):
    metrics = {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
    }
    return metrics

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts

def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)


def build_module_tree(module: nn.Module, module_name: str = "") -> dict:
    """
    Recursively traverse the given nn.Module and build a dictionary with:
    - 'name': module name
    - 'params': {parameter_name: number_of_elements}
    - 'children': {child_module_name: same dictionary structure}
    - 'total': total parameter count for this module (including all descendants)
    """
    # 1) Count direct parameters in this module
    direct_param_count = 0
    param_details = {}
    for pname, p in module.named_parameters(recurse=False):
        nump = p.numel()
        param_details[pname] = nump
        direct_param_count += nump

    # 2) Recursively process child modules
    children_info = {}
    for cname, child in module.named_children():
        children_info[cname] = build_module_tree(child, cname)

    # 3) Calculate total parameter count for this module (including all children)
    total = direct_param_count + sum(child["total"] for child in children_info.values())

    return {
        "name": module_name,
        "params": param_details,
        "children": children_info,
        "total": total,
    }


def print_module_tree(info: dict, parent_path: str = "", indent: int = 0):
    """
    Print the module tree built by build_module_tree() in a hierarchical format:
    "(total_parameter_count) (path_to_module_or_param)"
    The function sorts parameters and submodules in descending order of total size.
    """
    # Construct the current path
    name = info["name"]
    if not parent_path:
        full_path = name  # top level
    else:
        if name:  # submodule name is not empty
            full_path = f"{parent_path}/{name}"
        else:
            full_path = parent_path

    # Print total parameter count for the current module
    line = f"{info['total']:11,d} {full_path}"
    print(" " * indent + line)

    # Create a combined list of param_nodes (parameters) and child_nodes (submodules)
    param_nodes = []
    for param_name, count in info["params"].items():
        param_nodes.append({
            "name": param_name,
            "params": {},
            "children": {},
            "total": count,
        })

    child_nodes = list(info["children"].values())

    # Sort by 'total' in descending order
    combined = param_nodes + child_nodes
    combined.sort(key=lambda x: x["total"], reverse=True)

    # Recursively print all children
    for child_info in combined:
        print_module_tree(child_info, full_path, indent + 2)

def compute_rms(tensors):
    """Compute the root mean square (RMS) of a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.)
    return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)

def compute_global_norm(tensors):
    """Compute the global norm (L2 norm) across a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.)
    return torch.linalg.norm(flattened, ord=2)

def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x

def print_param_stats(model):
    """
    Prints formatted statistical information of the parameter values (not gradients)
    for the trainable parameters (.requires_grad=True) of the specified PyTorch model.

    - mean
    - std  (population standard deviation: std(unbiased=False))
    - L2 norm (param.data.norm())
    - RMS (root mean square: sqrt(mean(tensor^2)))

    The hierarchical name is displayed by replacing '.' with '/' in the default names
    (e.g., converting "layer.weight" to "layer/weight").
    """

    # List to temporarily store the statistics
    stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            mean_val = data.mean().item()
            std_val = data.std(unbiased=False).item()
            l2_val = data.norm().item()
            rms_val = data.pow(2).mean().sqrt().item()

            hierarchical_name = name.replace('.', '/')
            stats.append((hierarchical_name, mean_val, std_val, l2_val, rms_val))

    # Format and print the table with adjusted column widths
    # Format function to display numbers in scientific notation with 3 significant digits
    def fmt(v):
        return f"{v:.3e}"

    # Column width settings (adjust if necessary)
    col_widths = [60, 15, 15, 15, 15]
    header_format = f"{{:<{col_widths[0]}}}{{:>{col_widths[1]}}}{{:>{col_widths[2]}}}{{:>{col_widths[3]}}}{{:>{col_widths[4]}}}"
    row_format = header_format

    # Print the header
    print(header_format.format("Parameter", "Mean", "Std", "L2 norm", "RMS"))
    print("-" * (sum(col_widths) + 1))

    # Print the main content
    for hname, mean_val, std_val, l2_val, rms_val in stats:
        print(row_format.format(
            hname,
            fmt(mean_val),
            fmt(std_val),
            fmt(l2_val),
            fmt(rms_val),
        ))

def reshape(x, div):
    shape = x.shape
    B, L = shape[:2]
    trailing_dims = shape[2:]
    x_reshaped = x.view(B, L // div, div, *trailing_dims)
    permute_dims = (0, 2, 1) + tuple(range(3, x_reshaped.dim()))
    x_permuted = x_reshaped.permute(*permute_dims)
    final_shape = (B * div, L // div, *trailing_dims)
    y = x_permuted.contiguous().view(*final_shape)
    return y

def unreshape(y, div):
    shape = y.shape
    B_div, L_div = shape[:2]
    trailing_dims = shape[2:]
    B = B_div // div
    x_permuted_rev = y.view(B, div, L_div, *trailing_dims)
    permute_dims_rev = (0, 2, 1) + tuple(range(3, x_permuted_rev.dim()))
    x_reshaped_rev = x_permuted_rev.permute(*permute_dims_rev)
    final_shape = (B, L_div * div, *trailing_dims)
    x_original = x_reshaped_rev.contiguous().view(*final_shape)
    return x_original


def upload_to_s3(local_path: str, s3_bucket: str, s3_path: str) -> bool:
    """
    Upload every file under local_path to the specified S3 bucket/prefix.
    Credentials are loaded from the environment using python-dotenv.
    
    Returns True if upload succeeded, False otherwise.
    """
    load_dotenv()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url="https://s3-msk.tinkoff.ru",
        )
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                s3_key = os.path.join(s3_path, relative_path.replace("\\", "/"))
                s3_client.upload_file(local_file, s3_bucket, s3_key)
                print(f"[upload_to_s3] Uploaded {local_file} -> s3://{s3_bucket}/{s3_key}")
        return True
    except Exception as exc:
        print(f"[upload_to_s3] Error uploading files: {exc}")
        return False


def save_eval_video(
    video: np.ndarray,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
    fps: int = 16,
) -> None:
    """
    Save eval video to disk as MP4 and optionally upload to S3.
    
    Args:
        video: Video array of shape (B, T, H, W, C) with values in [0, 1] or [0, 255]
        logdir: Directory to save the video
        step: Training step for filename
        s3_bucket: S3 bucket name (None = no upload)
        s3_prefix: S3 prefix path for videos
        fps: Frames per second for the video
    
    S3 path format: s3://<bucket>/<prefix>/<wandb_run_name>/eval_step_<step>.mp4
    """
    try:
        import imageio
    except ImportError:
        print("[save_eval_video] imageio not installed, skipping video save")
        return
    
    # Ensure video is uint8
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    
    # video shape: (B, T, H, W, C) - concatenate batch dimension vertically
    B, T, H, W, C = video.shape
    video = video.transpose(1, 0, 2, 3, 4).reshape((T, B * H, W, C))
    
    # Create videos directory
    video_dir = logdir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video
    video_path = video_dir / f"viz_video_eval_step_{step}.mp4"
    imageio.mimwrite(str(video_path), video, fps=fps, codec='libx264', quality=8)
    print(f"[save_eval_video] Saved video to {video_path}")
    
    # Upload to S3 if bucket is specified
    if s3_bucket:
        # Get wandb run name
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        # Create staging directory with just the video
        staging_dir = logdir / "s3_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_video = staging_dir / f"viz_video_eval_step_{step}.mp4"
        shutil.copy2(video_path, staged_video)
        
        # Upload
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_eval_video] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        # Cleanup staging
        shutil.rmtree(staging_dir, ignore_errors=True)


def create_saliency_overlay_frame(
    image: np.ndarray,
    actor_saliency: np.ndarray,
    critic_saliency: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'hot'
) -> np.ndarray:
    """Create visualization overlay combining image with saliency maps.
    
    Creates a side-by-side view: [original | actor saliency | critic saliency]
    
    Args:
        image: Original image (H, W, C) in uint8 [0, 255]
        actor_saliency: Actor saliency (H, W) in [0, 1]
        critic_saliency: Critic saliency (H, W) in [0, 1]
        alpha: Blending factor for overlay
        colormap: Colormap name for saliency ('hot', 'jet', 'viridis')
        
    Returns:
        Combined visualization (H, 3*W, C) in uint8
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    H, W, C = image.shape
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap to saliency maps (returns RGBA in [0, 1])
    actor_colored = (cmap(actor_saliency)[..., :3] * 255).astype(np.uint8)
    critic_colored = (cmap(critic_saliency)[..., :3] * 255).astype(np.uint8)
    
    # Blend with original image
    image_float = image.astype(np.float32)
    actor_overlay = (1 - alpha) * image_float + alpha * actor_colored.astype(np.float32)
    critic_overlay = (1 - alpha) * image_float + alpha * critic_colored.astype(np.float32)
    
    actor_overlay = np.clip(actor_overlay, 0, 255).astype(np.uint8)
    critic_overlay = np.clip(critic_overlay, 0, 255).astype(np.uint8)
    
    # Concatenate horizontally: [original | actor | critic]
    combined = np.concatenate([image, actor_overlay, critic_overlay], axis=1)
    
    return combined


def save_saliency_video(
    video: np.ndarray,
    actor_saliency: np.ndarray,
    critic_saliency: np.ndarray,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
    fps: int = 16,
    alpha: float = 0.5,
    colormap: str = 'hot',
) -> None:
    """
    Save eval video with saliency overlays to disk and optionally upload to S3.
    
    Creates side-by-side video: [original | actor saliency | critic saliency]
    
    Args:
        video: Video array of shape (B, T, H, W, C) with values in [0, 1] or [0, 255]
        actor_saliency: Actor saliency array of shape (B, T, H, W) in [0, 1]
        critic_saliency: Critic saliency array of shape (B, T, H, W) in [0, 1]
        logdir: Directory to save the video
        step: Training step for filename
        s3_bucket: S3 bucket name (None = no upload)
        s3_prefix: S3 prefix path for videos
        fps: Frames per second for the video
        alpha: Blending factor for overlay
        colormap: Colormap name for saliency
    """
    try:
        import imageio
    except ImportError:
        print("[save_saliency_video] imageio not installed, skipping video save")
        return
    
    # Ensure video is uint8
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    
    B, T, H, W, C = video.shape
    
    # Create saliency overlay frames
    saliency_frames = []
    for b in range(B):
        for t in range(T):
            frame = create_saliency_overlay_frame(
                video[b, t],
                actor_saliency[b, t],
                critic_saliency[b, t],
                alpha=alpha,
                colormap=colormap
            )
            saliency_frames.append(frame)
    
    # Reshape: we had (B * T) frames, each of shape (H, 3*W, C)
    # For consistency with original video format, reshape to (T, B*H, 3*W, C)
    saliency_video = np.array(saliency_frames).reshape(B, T, H, 3 * W, C)
    saliency_video = saliency_video.transpose(1, 0, 2, 3, 4).reshape((T, B * H, 3 * W, C))
    
    # Create videos directory
    video_dir = logdir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video
    video_path = video_dir / f"viz_video_saliency_step_{step}.mp4"
    imageio.mimwrite(str(video_path), saliency_video, fps=fps, codec='libx264', quality=8)
    print(f"[save_saliency_video] Saved saliency video to {video_path}")
    
    # Upload to S3 if bucket is specified
    if s3_bucket:
        # Get wandb run name
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        # Create staging directory with just the video
        staging_dir = logdir / "s3_staging_saliency"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_video = staging_dir / f"viz_video_saliency_step_{step}.mp4"
        shutil.copy2(video_path, staged_video)
        
        # Upload
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_saliency_video] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        # Cleanup staging
        shutil.rmtree(staging_dir, ignore_errors=True)


def save_posthoc_decoder_video(
    original_video: np.ndarray,
    posterior_video: np.ndarray,
    imagination_video: np.ndarray,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
    fps: int = 16,
) -> None:
    """
    Save post-hoc decoder visualization video to disk and optionally upload to S3.
    
    Creates side-by-side video: [original | posterior reconstruction | imagination]
    
    Args:
        original_video: Ground truth video (B, T, H, W, C) in [0, 1] or [0, 255]
        posterior_video: Posterior reconstruction (B, T, H, W, C) in [0, 1]
        imagination_video: Imagined future (B, T_imag, H, W, C) in [0, 1]
        logdir: Directory to save the video
        step: Training step for filename
        s3_bucket: S3 bucket name (None = no upload)
        s3_prefix: S3 prefix path for videos
        fps: Frames per second for the video
    """
    try:
        import imageio
    except ImportError:
        print("[save_posthoc_decoder_video] imageio not installed, skipping video save")
        return
    
    def to_uint8(x):
        if np.issubdtype(x.dtype, np.floating):
            x = np.clip(255 * x, 0, 255).astype(np.uint8)
        return x
    
    original = to_uint8(original_video)
    posterior = to_uint8(posterior_video)
    imagination = to_uint8(imagination_video)
    
    B, T, H, W, C = original.shape
    _, T_imag, _, _, _ = imagination.shape
    
    # For the comparison video, we show:
    # - First T frames: original | posterior | black (no imagination yet)
    # - Then T_imag frames: black | black | imagination
    # Or simpler: concatenate posterior (T frames) + imagination (T_imag frames) side by side
    
    # Create frames for posterior comparison (original | posterior)
    posterior_frames = []
    for b in range(min(B, 1)):  # Just first episode
        for t in range(T):
            frame = np.concatenate([original[b, t], posterior[b, t]], axis=1)
            posterior_frames.append(frame)
    
    # Create frames for imagination (last original | imagination)
    imag_frames = []
    last_original = original[0, -1]
    for t in range(T_imag):
        frame = np.concatenate([last_original, imagination[0, t]], axis=1)
        imag_frames.append(frame)
    
    # Combine into single video: posterior comparison, then imagination
    all_frames = posterior_frames + imag_frames
    combined_video = np.stack(all_frames, axis=0)  # (T + T_imag, H, 2*W, C)
    
    # Create videos directory
    video_dir = logdir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video
    video_path = video_dir / f"viz_video_posthoc_recon_step_{step}.mp4"
    imageio.mimwrite(str(video_path), combined_video, fps=fps, codec='libx264', quality=8)
    print(f"[save_posthoc_decoder_video] Saved video to {video_path}")
    
    # Upload to S3 if bucket is specified
    if s3_bucket:
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        staging_dir = logdir / "s3_staging_posthoc"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_video = staging_dir / f"viz_video_posthoc_recon_step_{step}.mp4"
        shutil.copy2(video_path, staged_video)
        
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_posthoc_decoder_video] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        shutil.rmtree(staging_dir, ignore_errors=True)


def save_counterfactual_grid(
    init_image: np.ndarray,
    counterfactual_videos: list,
    action_labels: list,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
) -> None:
    """
    Save counterfactual comparison grid image.
    
    Shows initial state + multiple imagined futures from different actions.
    
    Args:
        init_image: Initial observation (H, W, C) in [0, 1] or [0, 255]
        counterfactual_videos: List of imagined futures [(T, H, W, C), ...]
        action_labels: Labels for each action sequence
        logdir: Directory to save the image
        step: Training step for filename
        s3_bucket: S3 bucket name (None = no upload)
        s3_prefix: S3 prefix path
    """
    try:
        import imageio
    except ImportError:
        print("[save_counterfactual_grid] imageio not installed, skipping")
        return
    
    def to_uint8(x):
        if np.issubdtype(x.dtype, np.floating):
            x = np.clip(255 * x, 0, 255).astype(np.uint8)
        return x
    
    init = to_uint8(init_image)
    H, W, C = init.shape
    
    num_futures = len(counterfactual_videos)
    T = counterfactual_videos[0].shape[0]
    
    # Create grid: rows = different action sequences, cols = timesteps
    # First column is initial state (repeated)
    grid_h = num_futures * H
    grid_w = (T + 1) * W
    
    grid = np.zeros((grid_h, grid_w, C), dtype=np.uint8)
    
    for i, renders in enumerate(counterfactual_videos):
        renders = to_uint8(renders)
        y_offset = i * H
        
        # Initial state
        grid[y_offset:y_offset+H, 0:W] = init
        
        # Future frames
        for t in range(T):
            x_offset = (t + 1) * W
            grid[y_offset:y_offset+H, x_offset:x_offset+W] = renders[t]
    
    # Create directory
    image_dir = logdir / "eval_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = image_dir / f"viz_image_counterfactual_step_{step}.png"
    imageio.imwrite(str(image_path), grid)
    print(f"[save_counterfactual_grid] Saved image to {image_path}")
    
    # Upload to S3
    if s3_bucket:
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        staging_dir = logdir / "s3_staging_cf"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_img = staging_dir / f"viz_image_counterfactual_step_{step}.png"
        shutil.copy2(image_path, staged_img)
        
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_counterfactual_grid] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        shutil.rmtree(staging_dir, ignore_errors=True)


def save_open_loop_prediction_video(
    context_true: np.ndarray,
    context_rendered: np.ndarray,
    future_true: np.ndarray,
    future_rendered: np.ndarray,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
    fps: int = 8,
) -> None:
    """
    Save open-loop prediction visualization as video.
    
    Creates a 2-row video:
    - Row 1 (True): Context frames → Future frames (ground truth)
    - Row 2 (Model): Context rendered → Future imagined
    
    Context frames are bordered in green, future frames in blue (true) / red (predicted).
    
    Args:
        context_true: Ground truth context frames (K, H, W, C)
        context_rendered: Rendered context from latents (K, H, W, C)
        future_true: Ground truth future frames (H_pred, H, W, C)
        future_rendered: Imagined future frames (H_pred, H, W, C)
        logdir: Directory to save
        step: Training step
        s3_bucket: S3 bucket (None = no upload)
        s3_prefix: S3 prefix
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("[save_open_loop_prediction_video] imageio not installed, skipping")
        return
    
    def to_uint8(x):
        if np.issubdtype(x.dtype, np.floating):
            x = np.clip(255 * x, 0, 255).astype(np.uint8)
        return x
    
    def add_border(img, color, width=3):
        """Add colored border to image."""
        img = img.copy()
        img[:width, :] = color
        img[-width:, :] = color
        img[:, :width] = color
        img[:, -width:] = color
        return img
    
    context_true = to_uint8(context_true)
    context_rendered = to_uint8(context_rendered)
    future_true = to_uint8(future_true)
    future_rendered = to_uint8(future_rendered)
    
    K = context_true.shape[0]
    H_pred = future_true.shape[0]
    img_H, img_W, C = context_true.shape[1:]
    
    # Colors for borders
    GREEN = np.array([0, 255, 0], dtype=np.uint8)  # Context
    BLUE = np.array([0, 100, 255], dtype=np.uint8)  # True future
    RED = np.array([255, 100, 0], dtype=np.uint8)   # Predicted future
    
    frames = []
    
    # Context phase (show both rows with context)
    for t in range(K):
        true_frame = add_border(context_true[t], GREEN)
        model_frame = add_border(context_rendered[t], GREEN)
        
        # Stack vertically: true on top, model on bottom
        combined = np.vstack([true_frame, model_frame])
        frames.append(combined)
    
    # Future phase (show ground truth vs prediction)
    for t in range(H_pred):
        true_frame = add_border(future_true[t], BLUE)
        model_frame = add_border(future_rendered[t], RED)
        
        combined = np.vstack([true_frame, model_frame])
        frames.append(combined)
    
    video = np.stack(frames, axis=0)
    
    # Create directory
    video_dir = logdir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video
    video_path = video_dir / f"viz_video_openloop_step_{step}.mp4"
    imageio.mimwrite(str(video_path), video, fps=fps, codec='libx264', quality=8)
    print(f"[save_open_loop_prediction_video] Saved video to {video_path}")
    
    # Upload to S3
    if s3_bucket:
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        staging_dir = logdir / "s3_staging_openloop"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_video = staging_dir / f"viz_video_openloop_step_{step}.mp4"
        shutil.copy2(video_path, staged_video)
        
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_open_loop_prediction_video] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        shutil.rmtree(staging_dir, ignore_errors=True)


def save_open_loop_prediction_grid(
    context_true: np.ndarray,
    context_rendered: np.ndarray,
    future_true: np.ndarray,
    future_rendered: np.ndarray,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
    max_frames: int = 20,
) -> None:
    """
    Save open-loop prediction as a grid image (like in papers).
    
    Creates a 2-row grid:
    - Row 1: True (context frames | vertical divider | future frames)
    - Row 2: Model (context rendered | vertical divider | future imagined)
    
    Args:
        context_true: Ground truth context frames (K, H, W, C)
        context_rendered: Rendered context from latents (K, H, W, C)
        future_true: Ground truth future frames (H_pred, H, W, C)
        future_rendered: Imagined future frames (H_pred, H, W, C)
        logdir: Directory to save
        step: Training step
        s3_bucket: S3 bucket
        s3_prefix: S3 prefix
        max_frames: Maximum frames per row (subsample if needed)
    """
    try:
        import imageio
    except ImportError:
        print("[save_open_loop_prediction_grid] imageio not installed, skipping")
        return
    
    def to_uint8(x):
        if np.issubdtype(x.dtype, np.floating):
            x = np.clip(255 * x, 0, 255).astype(np.uint8)
        return x
    
    context_true = to_uint8(context_true)
    context_rendered = to_uint8(context_rendered)
    future_true = to_uint8(future_true)
    future_rendered = to_uint8(future_rendered)
    
    K = context_true.shape[0]
    H_pred = future_true.shape[0]
    img_H, img_W, C = context_true.shape[1:]
    
    # Subsample if too many frames
    total_frames = K + H_pred
    if total_frames > max_frames:
        # Keep all context, subsample future
        future_indices = np.linspace(0, H_pred - 1, max_frames - K, dtype=int)
        future_true = future_true[future_indices]
        future_rendered = future_rendered[future_indices]
        H_pred = len(future_indices)
    
    # Divider width
    div_width = 4
    
    # Total width: K context + divider + H_pred future
    grid_w = K * img_W + div_width + H_pred * img_W
    grid_h = 2 * img_H + div_width  # 2 rows + horizontal divider
    
    grid = np.ones((grid_h, grid_w, C), dtype=np.uint8) * 128  # Gray background
    
    # Row 1: True
    y = 0
    for t in range(K):
        grid[y:y+img_H, t*img_W:(t+1)*img_W] = context_true[t]
    
    x_offset = K * img_W + div_width
    for t in range(H_pred):
        grid[y:y+img_H, x_offset+t*img_W:x_offset+(t+1)*img_W] = future_true[t]
    
    # Row 2: Model
    y = img_H + div_width
    for t in range(K):
        grid[y:y+img_H, t*img_W:(t+1)*img_W] = context_rendered[t]
    
    for t in range(H_pred):
        grid[y:y+img_H, x_offset+t*img_W:x_offset+(t+1)*img_W] = future_rendered[t]
    
    # Add labels (text would require PIL, so use colored markers instead)
    # Green marker for context, blue for true future, red for predicted future
    marker_h = 5
    GREEN = np.array([0, 255, 0], dtype=np.uint8)
    BLUE = np.array([0, 100, 255], dtype=np.uint8)
    RED = np.array([255, 100, 0], dtype=np.uint8)
    
    # Context markers (both rows)
    grid[0:marker_h, 0:K*img_W] = GREEN
    grid[img_H+div_width:img_H+div_width+marker_h, 0:K*img_W] = GREEN
    
    # Future markers
    grid[0:marker_h, x_offset:x_offset+H_pred*img_W] = BLUE  # True
    grid[img_H+div_width:img_H+div_width+marker_h, x_offset:x_offset+H_pred*img_W] = RED  # Predicted
    
    # Create directory
    image_dir = logdir / "eval_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = image_dir / f"viz_image_openloop_step_{step}.png"
    imageio.imwrite(str(image_path), grid)
    print(f"[save_open_loop_prediction_grid] Saved image to {image_path}")
    
    # Upload to S3
    if s3_bucket:
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        staging_dir = logdir / "s3_staging_openloop_grid"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_img = staging_dir / f"viz_image_openloop_step_{step}.png"
        shutil.copy2(image_path, staged_img)
        
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_open_loop_prediction_grid] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        shutil.rmtree(staging_dir, ignore_errors=True)


def save_uncertainty_visualization(
    context_true: np.ndarray,
    future_true: np.ndarray,
    future_samples: list,
    logdir: pathlib.Path,
    step: int,
    s3_bucket: str = None,
    s3_prefix: str = "ne_dreamer",
    max_frames: int = 15,
) -> None:
    """
    Save uncertainty visualization showing multiple predicted futures.
    
    Creates a grid with:
    - Row 1: Ground truth (context | future)
    - Rows 2+: Multiple predicted samples (context same | different futures)
    
    Args:
        context_true: Context frames (K, H, W, C)
        future_true: True future frames (H_pred, H, W, C)
        future_samples: List of predicted futures [(H_pred, H, W, C), ...]
        logdir: Directory to save
        step: Training step
        s3_bucket: S3 bucket
        s3_prefix: S3 prefix
        max_frames: Max frames per row
    """
    try:
        import imageio
    except ImportError:
        print("[save_uncertainty_visualization] imageio not installed, skipping")
        return
    
    def to_uint8(x):
        if np.issubdtype(x.dtype, np.floating):
            x = np.clip(255 * x, 0, 255).astype(np.uint8)
        return x
    
    context_true = to_uint8(context_true)
    future_true = to_uint8(future_true)
    future_samples = [to_uint8(s) for s in future_samples]
    
    K = context_true.shape[0]
    H_pred = future_true.shape[0]
    img_H, img_W, C = context_true.shape[1:]
    num_samples = len(future_samples)
    
    # Subsample futures if needed
    total_frames = K + H_pred
    if total_frames > max_frames:
        future_indices = np.linspace(0, H_pred - 1, max_frames - K, dtype=int)
        future_true = future_true[future_indices]
        future_samples = [s[future_indices] for s in future_samples]
        H_pred = len(future_indices)
    
    div_width = 4
    grid_w = K * img_W + div_width + H_pred * img_W
    grid_h = (1 + num_samples) * img_H + num_samples * div_width
    
    grid = np.ones((grid_h, grid_w, C), dtype=np.uint8) * 128
    
    # Row 1: Ground truth
    y = 0
    for t in range(K):
        grid[y:y+img_H, t*img_W:(t+1)*img_W] = context_true[t]
    
    x_offset = K * img_W + div_width
    for t in range(H_pred):
        grid[y:y+img_H, x_offset+t*img_W:x_offset+(t+1)*img_W] = future_true[t]
    
    # Rows 2+: Samples
    for s_idx, sample in enumerate(future_samples):
        y = (s_idx + 1) * img_H + (s_idx + 1) * div_width
        
        # Context (same for all samples)
        for t in range(K):
            grid[y:y+img_H, t*img_W:(t+1)*img_W] = context_true[t]
        
        # Predicted future
        for t in range(H_pred):
            grid[y:y+img_H, x_offset+t*img_W:x_offset+(t+1)*img_W] = sample[t]
    
    # Create directory
    image_dir = logdir / "eval_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = image_dir / f"viz_image_uncertainty_step_{step}.png"
    imageio.imwrite(str(image_path), grid)
    print(f"[save_uncertainty_visualization] Saved image to {image_path}")
    
    # Upload to S3
    if s3_bucket:
        run_name = "default"
        if wandb.run is not None:
            run_name = wandb.run.name or wandb.run.id or "default"
        
        s3_path = f"{s3_prefix}/{run_name}"
        
        staging_dir = logdir / "s3_staging_uncertainty"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_img = staging_dir / f"viz_image_uncertainty_step_{step}.png"
        shutil.copy2(image_path, staged_img)
        
        success = upload_to_s3(str(staging_dir), s3_bucket, s3_path)
        if success:
            print(f"[save_uncertainty_visualization] Uploaded to s3://{s3_bucket}/{s3_path}")
        
        shutil.rmtree(staging_dir, ignore_errors=True)
