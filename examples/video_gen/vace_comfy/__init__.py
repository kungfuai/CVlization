# VACE ComfyUI Implementation
# Video Animation Control Extension for ComfyUI

from .nodes_wan import WanVaceToVideo, CreateFadeMaskAdvanced
from .nodes_model_advanced import ModelSamplingSD3, CFGZeroStar, UNetTemporalAttentionMultiply, SkipLayerGuidanceDiT

__version__ = "1.0.0"
__author__ = "CVlization"
__description__ = "VACE (Video Animation Control Extension) implementation for ComfyUI" 