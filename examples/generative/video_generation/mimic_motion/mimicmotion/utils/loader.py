import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline
from .hf_cache import resolve_asset_path

logger = logging.getLogger(__name__)

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", torch_dtype=torch.float16, variant="fp16")
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])

def create_pipeline(infer_config, device):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str): 
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """
    ckpt_path = getattr(infer_config, "ckpt_path", None)
    if ckpt_path is None and hasattr(infer_config, "ckpt_repo_id"):
        ckpt_filename = getattr(infer_config, "ckpt_filename", "MimicMotion_1-1.pth")
        ckpt_path = f"hf://{infer_config.ckpt_repo_id}/{ckpt_filename}"
    if ckpt_path is None:
        raise ValueError("Checkpoint path not provided in inference config.")

    resolved_ckpt = resolve_asset_path(
        ckpt_path, env_override="MIMICMOTION_CKPT_PATH"
    )

    mimicmotion_models = MimicMotionModel(infer_config.base_model_path)
    mimicmotion_models.load_state_dict(
        torch.load(resolved_ckpt, map_location="cpu"), strict=False
    )
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae, 
        image_encoder=mimicmotion_models.image_encoder, 
        unet=mimicmotion_models.unet, 
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=mimicmotion_models.pose_net
    )
    return pipeline
