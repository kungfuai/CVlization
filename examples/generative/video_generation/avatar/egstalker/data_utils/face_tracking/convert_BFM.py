import numpy as np
from scipy.io import loadmat
import os
from pathlib import Path
import sys

# Auto-download BFM model if not present
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from bfm_model import download_bfm_model, BFM_CACHE_DIR

    # Ensure BFM model is downloaded
    bfm_cache = download_bfm_model(download_all=True)

    # Create symlink from 3DMM to cache if needed
    dmm_dir = Path("3DMM")
    if not dmm_dir.exists():
        print(f"Creating symlink: 3DMM -> {bfm_cache}")
        dmm_dir.symlink_to(bfm_cache)
except ImportError:
    print("Warning: Could not import BFM download utility. Looking for files in 3DMM directory...")

original_BFM = loadmat("3DMM/01_MorphableModel.mat")

# Check for topology_info.npy
if not os.path.exists("3DMM/topology_info.npy"):
    raise FileNotFoundError(
        "3DMM/topology_info.npy not found. This file must be generated or obtained from the original EGSTalker repository."
    )

sub_inds = np.load("3DMM/topology_info.npy", allow_pickle=True).item()["sub_inds"]

shapePC = original_BFM["shapePC"]
shapeEV = original_BFM["shapeEV"]
shapeMU = original_BFM["shapeMU"]
texPC = original_BFM["texPC"]
texEV = original_BFM["texEV"]
texMU = original_BFM["texMU"]

b_shape = shapePC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_shape = shapeMU.reshape(-1, 3)

b_tex = texPC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_tex = texMU.reshape(-1, 3)

b_shape = b_shape[:, sub_inds, :].reshape(199, -1)
mu_shape = mu_shape[sub_inds, :].reshape(-1)
b_tex = b_tex[:, sub_inds, :].reshape(199, -1)
mu_tex = mu_tex[sub_inds, :].reshape(-1)

exp_info = np.load("3DMM/exp_info.npy", allow_pickle=True).item()
np.save(
    "3DMM/3DMM_info.npy",
    {
        "mu_shape": mu_shape,
        "b_shape": b_shape,
        "sig_shape": shapeEV.reshape(-1),
        "mu_exp": exp_info["mu_exp"],
        "b_exp": exp_info["base_exp"],
        "sig_exp": exp_info["sig_exp"],
        "mu_tex": mu_tex,
        "b_tex": b_tex,
        "sig_tex": texEV.reshape(-1),
    },
)
