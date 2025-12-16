import os
from dataclasses import dataclass

from omegaconf import OmegaConf

PATH_TO_YOUR_MODEL = "zai-org/RealVideo/model.pt"  # Replace with your model path


@dataclass
class AudioConfig:
    sample_rate: int = 16000


@dataclass
class VideoConfig:
    fps: int = 16

    frame_width: int = 480
    frame_height: int = 640

    speaking_prompt: str = "A character is talking."
    silence_prompt: str = "A character is looking at the camera."


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8003
    diffusion_socket_port: int = 9090
    app_socket_port: int = 9091
    app_ready_socket_port: int = 9092
    diffusion_ready_socket_port: int = 9093


@dataclass
class LipSyncConfig:
    fps: int = 16

    s2v_segment_latent_length = 80

    self_forcing_config_path: str = (
        "self_forcing/configs/sample_14B_s2v_sparse_nfb2.yaml"
    )
    # self_forcing_config_path: str = 'self_forcing/configs/sample_14B_s2v_sparse_nfb2_2steps.yaml'

    checkpoint_path: str = PATH_TO_YOUR_MODEL

    audio_padding_div = 16
    audio_padding_rem = 0
    audio_min_length = 16
    audio_segment_length = 80
    s2v_video_refresh_interval = 20
    compile = True
    profile = True
    fp8_quantize = False
    no_refresh_inference = True

    dit_config = OmegaConf.load(self_forcing_config_path)
    default_config = OmegaConf.load("self_forcing/configs/default_config.yaml")
    dit_config = OmegaConf.merge(default_config, dit_config)


class Config:
    def __init__(self):
        self.audio = AudioConfig()
        self.video = VideoConfig()
        self.server = ServerConfig()
        self.lip_sync = LipSyncConfig()

        self._load_from_env()

    def _load_from_env(self):
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        self.log_level = os.getenv("LOG_LEVEL", "DEBUG")
        self.self_focing_config_path = os.getenv("CONFIG_PATH", "")
        self.audio_samples_per_video_block = round(
            self.audio.sample_rate
            / self.video.fps
            * self.lip_sync.dit_config.num_frame_per_block
            * 4
        )  # in audio samples, (4 for vae)
        self.lip_sync.audio_min_length = (
            4 * self.lip_sync.dit_config.num_frame_per_block
        )  # in frames, (4 for vae)


config = Config()
