import types
from pyexpat import features

from typing import List, Optional
import torch
from torch import nn
from .scheduler import SchedulerInterface, FlowMatchScheduler

try:

    from ..wan.modules.tokenizers import HuggingfaceTokenizer
    from ..wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
    from ..wan.modules.vae import _video_vae
    from ..wan.modules.t5 import umt5_xxl
    from ..wan.modules.clip import CLIPModel
    from ..wan.modules.clip import clip_xlm_roberta_vit_h_14
    from ..wan.modules.causal_model import CausalWanModel
    from ..wan.modules.model_s2v import WanModel_S2V
    from ..wan.modules.causal_model_s2v import CausalWanModel_S2V

except ImportError:
    from wan.modules.tokenizers import HuggingfaceTokenizer
    from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
    from wan.modules.vae import _video_vae
    from wan.modules.t5 import umt5_xxl
    from wan.modules.clip import CLIPModel
    from wan.modules.clip import clip_xlm_roberta_vit_h_14
    from wan.modules.causal_model import CausalWanModel
    from wan.modules.model_s2v import WanModel_S2V
    from wan.modules.causal_model_s2v import CausalWanModel_S2V

import torch.nn.functional as F
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput
from transformers import Wav2Vec2FeatureExtractor
import os
import math
import numpy as np
from einops import rearrange

class Wav2VecModel(Wav2Vec2Model):
    """
    Wav2VecModel is a custom model class that extends the Wav2Vec2Model class from the transformers library.
    It inherits all the functionality of the Wav2Vec2Model and adds additional methods for feature extraction and encoding.
    ...

    Attributes:
        base_model (Wav2Vec2Model): The base Wav2Vec2Model object.

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            Forward pass of the Wav2VecModel.
            It takes input_values, seq_len, and other optional parameters as input and returns the output of the base model.

        feature_extract(input_values, seq_len):
            Extracts features from the input_values using the base model.

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            Encodes the extracted features using the base model and returns the encoded features.
    """
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the Wav2Vec model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the Wav2Vec model.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        """
        Extracts features from the input values and returns the extracted features.

        Parameters:
        input_values (torch.Tensor): The input values to be processed.
        seq_len (torch.Tensor): The sequence lengths of the input values.

        Returns:
        extracted_features (torch.Tensor): The extracted features from the input values.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Encodes the input features into the output space.

        Args:
            extract_features (torch.Tensor): The extracted features from the audio signal.
            attention_mask (torch.Tensor, optional): Attention mask to be used for padding.
            mask_time_indices (torch.Tensor, optional): Masked indices for the time dimension.
            output_attentions (bool, optional): If set to True, returns the attention weights.
            output_hidden_states (bool, optional): If set to True, returns all hidden states.
            return_dict (bool, optional): If set to True, returns a BaseModelOutput instead of the tuple.

        Returns:
            The encoded output features.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def linear_interpolation(features, seq_len):
    """
    Transpose the features to interpolate linearly.

    Args:
        features (torch.Tensor): The extracted features to be interpolated.
        seq_len (torch.Tensor): The sequence lengths of the features.

    Returns:
        torch.Tensor: The interpolated features.
    """
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def process_audio_emb(audio_emb):
    """
    处理音频嵌入数据的函数，通过创建滑动窗口来增强音频序列的上下文信息
    参数:
        audio_emb (torch.Tensor): 输入的音频嵌入张量，形状为(batch_size, seq_len, f, embed_dim)
            batch_size: 批次大小
            seq_len: 序列长度
            f: 频域特征数
            embed_dim: 嵌入维度
    返回:
        torch.Tensor: 处理后的音频嵌入张量，形状与输入相同
    """
    # 获取输入张量的各个维度大小
    batch_size, seq_len, f, embed_dim = audio_emb.shape

    indices = torch.arange(seq_len).unsqueeze(1) + torch.arange(-2, 3).unsqueeze(0)
    indices = indices.clamp(0, seq_len - 1)

    audio_emb = audio_emb[:, indices]

    return audio_emb

class WanAudioEncoder(torch.nn.Module):
    def __init__(self,
                 wav2vec_model_path='facebook/wav2vec2-base-960h',
                 only_last_features=False,
                  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path).to(device=torch.device('cuda'))
        self.audio_encoder.eval()
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.only_last_features = only_last_features
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_model_path, local_files_only=True)

    def get_embedding(self, speech_array, fps: float = 16, sampling_rate = 16000):
        """preprocess wav audio file convert to embeddings

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        speech_array = speech_array.float()
        audio_feature = self.wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values[0]
        seq_len = math.ceil(len(audio_feature[0]) / sampling_rate * fps)

        audio_feature = torch.from_numpy(
            audio_feature).float().to(device=self.device)
        if len(audio_feature.shape) == 1:
            audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=seq_len, output_hidden_states=True)
        assert len(embeddings) > 0, "Fail to extract audio embedding"

        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1)
            audio_emb = rearrange(audio_emb, "a b s d -> a s b d")

        audio_emb = audio_emb.detach()

        return audio_emb

    def forward(self, audio_prompts, fps: float = 16):
        if type(audio_prompts) == list:
            if len(audio_prompts[0].shape) == 1:
                audio_prompts = torch.stack(audio_prompts, dim=0)
            else:
                audio_prompts = torch.concat(audio_prompts, dim=0)
        embedding = self.get_embedding(audio_prompts.float(), fps)
        embedding = process_audio_emb(embedding)
        return {"audio_emb":embedding}

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

class WanImageEncoder(torch.nn.Module):
    def __init__(self, dtype=torch.float16, device='cpu', checkpoint_path=None):
        super().__init__()
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype,
            device='cpu')
        self.model = self.model.eval().requires_grad_(False)
        print(f'loading {checkpoint_path}')
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location='cpu'))
        self.model.to('cuda')


    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    @torch.no_grad()
    def forward(self, videos):
        # preprocess
        videos = videos.squeeze(2) # b, c, t, h, w -> b, c, h, w
        size = (self.model.image_size,) * 2
        videos = F.interpolate(
                    videos,
                    size=size,
                    mode='bicubic',
                    align_corners=False)
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out


class WanTextEncoder(torch.nn.Module):
    def __init__(self, seq_len=512) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        self.seq_len = 512

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.2-S2V-14B/google/umt5-xxl/", seq_len=self.seq_len, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    @torch.no_grad()
    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }

class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    @torch.no_grad()
    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            is_sparse_causal=False,
            local_attn_size=-1,
            sink_size=0,
            independent_first_frame=False,
            num_frame_per_block=1,
            low_cpu_mem_usage=True,
            skip_init_model=False,
            add_cls_branch=False,
            concat_time_embeddings=False,
    ):
        super().__init__()

        causal_model_kwargs = {'sink_size': sink_size, 'independent_first_frame': independent_first_frame, 'num_frame_per_block': num_frame_per_block, 'local_attn_size': local_attn_size}

        if is_causal or is_sparse_causal:
            if 'S2V' in model_name:
                model_cls = CausalWanModel_S2V
            else:
                model_cls = CausalWanModel
            if skip_init_model:
                pipeline_name = f"wan_models/{model_name}/"
                self.model = model_cls.from_config(pipeline_name, is_sparse=is_sparse_causal, **causal_model_kwargs)
            else:
                self.model, loading_info = model_cls.from_pretrained(
                    f"wan_models/{model_name}/",
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    is_sparse=is_sparse_causal,
                    output_loading_info=True,
                    **causal_model_kwargs)
        else:
            if 'S2V' in model_name:
                model_cls = WanModel_S2V
            else:
                model_cls = WanModel
            if skip_init_model:
                pipeline_name = f"wan_models/{model_name}/"
                self.model = model_cls.from_config(pipeline_name)
            else:
                self.model, loading_info = model_cls.from_pretrained(
                    f"wan_models/{model_name}/",
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    output_loading_info=True)
        if not skip_init_model and torch.distributed.get_rank() == 0:
            print(f"load model from {model_name}\n missing_keys: {loading_info['missing_keys']}\n unexpected_keys: {loading_info['unexpected_keys']}\n mismatched_keys: {loading_info['mismatched_keys']}\n error_msgs: {loading_info['error_msgs']}")

        self.model_type = self.model.model_type

        # add classify branch for DMD2
        if add_cls_branch:
            print('Add classify branch for fake_score.')
            atten_dim = self.model.dim
            time_embed_dim = self.model.dim
            self.concat_time_embeddings = concat_time_embeddings
            self.adding_cls_branch(
                atten_dim=atten_dim, num_class=1,
                time_embed_dim=time_embed_dim if self.concat_time_embeddings else 0)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal and not is_sparse_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 70000  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # Only tested for WAN2.1-T2V-1.3B & 14B, WAN2.1-I2V-1.3B & 14B
        self.model._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(1536, num_class)
        )
        self.model._cls_pred_branch.requires_grad_(True)
        disc_params = sum([p.numel() for p in self.model._cls_pred_branch.parameters() if p.requires_grad])
        num_registers = 3
        self.model._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self.model._register_tokens.requires_grad_(True)
        disc_params += sum([p.numel() for p in self.model._register_tokens.parameters() if p.requires_grad])

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock(dim=atten_dim, num_heads=atten_dim//128)
            gan_ca_blocks.append(block)
        self.model._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self.model._gan_ca_blocks.requires_grad_(True)
        disc_params += sum([p.numel() for p in self.model._gan_ca_blocks.parameters() if p.requires_grad])
        print(f'Discriminator parameter number: {disc_params}')

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        sp_dim: Optional[str] = None,
        sink_size = None,
        disable_float_conversion = False,
        **kwargs
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        if "audio_emb" in conditional_dict:
            prompt_embeds = conditional_dict
        if 'motion_latents' in conditional_dict:
            kwargs['motion_latents'] = conditional_dict['motion_latents']
        if 'drop_motion_frames' in conditional_dict:
            kwargs['drop_motion_frames'] = conditional_dict['drop_motion_frames']
        if 'audio_input' in conditional_dict:
            kwargs['audio_input'] = conditional_dict['audio_input']
        if 'ref_latents' in conditional_dict:
            kwargs['ref_latents'] = conditional_dict['ref_latents']
        if 'motion_frames' in conditional_dict:
            kwargs['motion_frames'] = conditional_dict['motion_frames']
        if sink_size is not None:
            kwargs['sink_size'] = sink_size

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        if noisy_image_or_video is None:
            noisy_input = None

        else:
            noisy_image_or_video = noisy_image_or_video.to(torch.bfloat16)
            noisy_input = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        # X0 prediction
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if kv_cache is not None:
                flow_pred = self.model(
                    noisy_input,
                    t=input_timestep,
                    context=prompt_embeds,
                    y=conditional_dict.get('image_latent', None),
                    clip_feature=conditional_dict.get('image_clip_features', None),
                    seq_len=self.seq_len,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    cache_start=cache_start,
                    sp_dim=sp_dim,
                    **kwargs
                )
                if isinstance(flow_pred, int):
                    return flow_pred
                flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
            else:
                if clean_x is not None:
                    # teacher forcing
                    flow_pred = self.model(
                        noisy_input,
                        t=input_timestep,
                        context=prompt_embeds,
                        y=conditional_dict.get('image_latent', None),
                        clip_feature=conditional_dict.get('image_clip_features', None),
                        seq_len=self.seq_len,
                        clean_x=clean_x.permute(0, 2, 1, 3, 4),
                        aug_t=aug_t,
                        sp_dim=sp_dim,
                        **kwargs
                    ).permute(0, 2, 1, 3, 4)
                else:
                    if classify_mode:
                        flow_pred, logits = self.model(
                            noisy_input,
                            t=input_timestep,
                            context=prompt_embeds,
                            y=conditional_dict.get('image_latent', None),
                            clip_feature=conditional_dict.get('image_clip_features', None),
                            seq_len=self.seq_len,
                            classify_mode=True,
                            register_tokens=self.model._register_tokens,
                            cls_pred_branch=self.model._cls_pred_branch,
                            gan_ca_blocks=self.model._gan_ca_blocks,
                            concat_time_embeddings=concat_time_embeddings,
                            sp_dim=sp_dim,
                            **kwargs
                        )
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                    else:
                        flow_pred = self.model(
                            noisy_input,
                            t=input_timestep,
                            context=prompt_embeds,
                            y=conditional_dict.get('image_latent', None),
                            clip_feature=conditional_dict.get('image_clip_features', None),
                            seq_len=self.seq_len,
                            sp_dim=sp_dim,
                            **kwargs
                        ).permute(0, 2, 1, 3, 4)

        if disable_float_conversion:
            pred_x0 = noisy_image_or_video - timestep[:,:,None,None,None]/1000 * flow_pred
        else:
            pred_x0 = noisy_image_or_video.float() - timestep[:,:,None,None,None]/1000 * flow_pred.float()
        # pred_x0 = self._convert_flow_pred_to_x0(
        #     flow_pred=flow_pred.flatten(0, 1),
        #     xt=noisy_image_or_video.flatten(0, 1),
        #     timestep=timestep.flatten(0, 1)
        # ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
