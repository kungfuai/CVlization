# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

import random
from typing import List, Tuple, Optional, Dict
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from data.data_utils import (
    create_sparse_mask,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_interpolate_video,
    get_flattened_position_ids_extrapolate_video,
)
from .qwen2_navit import NaiveCache, Qwen2ForCausalLM
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding3D

from config.config_factory import TrainingArguments
from common.utils.misc import AutoEncoderParams
from common.utils.distributed import get_global_rank
from common.utils.logging import get_logger
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from modeling.qwen2 import Qwen2Tokenizer
from common.val.utils import map_splits_to_samples, make_packed_vit_token_embed, uncond_split_pro
from data.common import shift_position_ids
from copy import deepcopy

class LanceConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config: AutoEncoderParams = None,
        latent_patch_size=(1, 2, 2),  # pt ph pw
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_num_frames = kwargs.get("max_num_frames", 25)  # TODO: 完善下
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class Lance(PreTrainedModel):
    config_class = LanceConfig
    base_model_prefix = "lance"

    def __init__(
        self,
        language_model: Qwen2ForCausalLM,
        vit_model: Qwen2_5_VisionTransformerPretrainedModel,
        vit_type: str = "qwen2_5_vl",
        config: LanceConfig = None,
        **kwargs
    ):
        super().__init__(config)
        self.language_model: Qwen2ForCausalLM = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self.logger = get_logger()
        self.log_rank0 = self.logger.info if get_global_rank() == 0 else lambda x: None
        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample_spatial = config.vae_config.downsample_spatial * config.latent_patch_size[-1]
            self.latent_downsample_temporal = config.vae_config.downsample_temporal
            self.max_num_latent_frames = config.max_num_frames // self.latent_downsample_temporal + 1
            self.latent_channel = config.vae_config.z_channels
            self.max_latent_size = config.max_latent_size
            self.patch_latent_dim = self.latent_patch_size[0] * self.latent_patch_size[1] * self.latent_patch_size[2] * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)

            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)  # vision input
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)  # vision ouput

            self.latent_pos_embed = PositionEmbedding3D(self.max_num_latent_frames, self.max_latent_size, self.hidden_size)

            safety = 1024 # 由于video前有文本，因此video的起始坐标不再是[0,0,0], 而是[n,n,n], 预留出安全位移
            self.pos_shift = self.max_latent_size * self.max_latent_size * self.max_num_latent_frames + safety

        if config.visual_und:
            self.vit_model: Qwen2_5_VisionTransformerPretrainedModel = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_type = vit_type
            if self.vit_type == "qwen2_5_vl":
                self.vit_hidden_size: int = config.vit_config.out_hidden_size
                self.connector: MLPconnector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)

            elif self.vit_type == "qwen_2_5_vl_original":
                pass # 注意⚠️ ViT encoder中的merger就是MLP connector了，已经包含MLP connector了
            else:
                raise ValueError(f"vit_model_type {self.vit_type} not supported")

            self.vit_model.eval()

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self.training_args: TrainingArguments = kwargs.get("training_args") # for 消融实验

        # for task embedding
        if self.training_args.use_task_embedding:
            self.num_tasks = 10 # 设置为一个大于实际task/modality 数量的值
            self.task_embedding = nn.Embedding(self.num_tasks, config.vit_config.out_hidden_size)

        if self.training_args.use_modality_embedding:
            self.num_modalities = 10 # 设置为一个大于实际task/modality 数量的值
            self.modality_embedding = nn.Embedding(self.num_modalities, config.vit_config.out_hidden_size)

    def update_tokenizer(self, tokenizer):
        self.tokenizer: Qwen2Tokenizer = tokenizer # add for debug, check, loss
        self.vocab_size_efficient = len(tokenizer) # 有效的词表长度，用于计算loss

    def process_attention_mask(self, current_attn_modes, current_split_lens, current_seq_len, device, BLOCK_SIZE=128):
        current_attn_modes_ = ["full" if mode_ in ["full_noise", "full_noise_target"] else mode_ for mode_ in current_attn_modes]
        sparse_mask = create_sparse_mask(current_seq_len, current_split_lens, current_attn_modes_, device)
        current_seq_len_sum = sum(current_seq_len)
        attention_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=current_seq_len_sum, KV_LEN=current_seq_len_sum, device=device, BLOCK_SIZE=BLOCK_SIZE, _compile=False
            )
        return attention_mask

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        sample_type: List[str],
        sample_N_target: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        vit_video_grid_thw: Optional[torch.IntTensor] = None,
        vae_video_grid_thw: Optional[torch.IntTensor] = None,
        video_grid_thw: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        vit_data_mode: Optional[List[str]] = None, # 指示每个vit_split online or offline
        key_frame_mask: Optional[torch.BoolTensor] = None,
        sample_task: Optional[torch.LongTensor] = None,
        sample_modality: Optional[torch.LongTensor] = None,
        BLOCK_SIZE: int = 128,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        # 获取其他参数信息
        N_vit_split = attn_modes.count("full")
        device = packed_text_ids.device
        apply_qwen_2_5_vl_pos_emb = getattr(self.training_args, "apply_qwen_2_5_vl_pos_emb", False)
        choose_frame_loss = random.random() > 0.5 # 50% 的概率执行预测视频
        sample_splits = map_splits_to_samples(sample_lens, split_lens)

        if apply_qwen_2_5_vl_pos_emb:  # TODO :

            packed_position_ids = []
            sample_lens_tensor = torch.tensor(sample_lens, device=device, dtype=torch.long)  # sample_lens是已经是padding的，即便为0，也会pad
            cu_sample_lens = torch.cat([torch.zeros(1, device=device, dtype=torch.long), sample_lens_tensor.cumsum(0)[:-1]])
            # i_num_grid_thw = 0
            for i_sample in range(len(sample_lens) - 1):
                text_ids = packed_text_ids[cu_sample_lens[i_sample] : cu_sample_lens[i_sample + 1]]
                left, right = sample_splits[i_sample][0], sample_splits[i_sample][-1] + 1
                grid_thw_rope  = video_grid_thw[i_sample]

                i_sample_task = sample_task[cu_sample_lens[i_sample] : cu_sample_lens[i_sample + 1]]
                i_sample_modality = sample_modality[cu_sample_lens[i_sample] : cu_sample_lens[i_sample + 1]]

                current_packed_position_ids, rope_deltas = self.language_model.get_rope_index(
                    input_ids=text_ids.unsqueeze(0),
                    image_grid_thw=grid_thw_rope,  # [[1,16,16]],
                    video_grid_thw=grid_thw_rope,  # video_grid_thw,
                    second_per_grid_ts=[1.0]*len(grid_thw_rope),  # second_per_grid_ts, 理论上应为1/6，实际使用 1
                    attention_mask=torch.ones([1, len(text_ids)], dtype=torch.long, device=device),  # attention_mask, 全1掩码？
                )
                # mrope 上区分 ref image vae特征与video vae特征
                current_packed_position_ids = shift_position_ids(current_packed_position_ids, pos_shift = 1000, attn_modes = attn_modes[left:right], split_lens = split_lens[left:right], shift_attn_mode=['full_noise',"full"], pro_type = 10, i_sample_task=i_sample_task, i_sample_modality=i_sample_modality)
                packed_position_ids.append(current_packed_position_ids)
            packed_position_ids = torch.cat(packed_position_ids, dim=-1)  # [3,1,sequence_length]

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))  # 构造模型的输入 L * C
        packed_sequence[packed_text_indexes] = packed_text_embedding[packed_text_indexes]

        if nested_attention_masks is None: # 走这里
            attn_modes_ = ["full" if mode=="full_noise" else mode for mode in attn_modes]
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes_, packed_text_embedding.device)
            seqlen = sum(sample_lens)  # 始终是max_num_tokens
            attention_mask = create_block_mask(sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, device=packed_text_embedding.device, BLOCK_SIZE=BLOCK_SIZE, _compile=True)
        else:
            attention_mask = nested_attention_masks

        if N_vit_split > 0: # 肥understanding 分支也可能有 VIT 条件，例如TI2I
            if self.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
                # NOTE video understanding部分
                with torch.no_grad():
                    packed_vit_token_embed = make_packed_vit_token_embed(packed_vit_tokens, vit_data_mode, vit_video_grid_thw, self.vit_model)
                if self.vit_type == "qwen2_5_vl":
                    packed_vit_token_embed = self.connector(packed_vit_token_embed) # L//4 x 2048 -> L//4 x 1536
                packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed  # NOTE: 这里替换真实的vit token embed！

        # 计算flow matching loss
        if self.config.visual_gen:
            pt, ph, pw = self.latent_patch_size  # e.g., 1, 1, 1
            packed_latent = []
            # NOTE: patchified_vae_latent_shapes 已经是计算了patchify的
            for latent, (t, h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                # NOTE: 这里的patchify是直接reshape的 (space2channel)，后面会通过 self.vae2llm(packed_latenat) 做映射;
                # NOTE: padded_latent的shape是被补齐的，取的最大的shape
                # TODO: 当前并不支持native shape；后续需要支持native shape

                # 假设 latent 形状是 [t*pt, h*ph, w*pw, c]，并且外层循环中你已经拿到了 t, h, w, pt, ph, pw
                patches = rearrange(latent, "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)", t=t, pt=pt, h=h, ph=ph, w=w, pw=pw)
                # patches 的形状就是 [t*h*w, pt*ph*pw*c]
                packed_latent.append(patches)
            packed_latent_clean = torch.cat(packed_latent, dim=0)  # (B*t*h*w, pt*ph*pw*c) -> e.g., (B*256, 48) for seedance vae with 256x256

            noise = torch.randn_like(packed_latent_clean)
            if getattr(self.training_args, "incre_time_pro", 0) <=0:
                packed_timesteps = torch.sigmoid(packed_timesteps)  # NOTE: 这里归一化到 [0, 1] 了。Gaussian + Sigmoid = Logit-Normal Distribution
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            if key_frame_mask != []:
                if not choose_frame_loss:
                    time_idx = torch.arange(len(packed_timesteps), device=packed_timesteps.device)[packed_timesteps > 0][key_frame_mask==1]
                    packed_timesteps[time_idx] = 0 # 将关键帧置为 clean
                    mse_loss_indexes = mse_loss_indexes[key_frame_mask==0] # 只在非关键帧上计算loss
                else:
                    mse_loss_indexes = mse_loss_indexes[key_frame_mask==1]

            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps) # [L, C]
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb

            packed_sequence[packed_vae_token_indexes] = packed_latent.to(packed_sequence.dtype) # NOTE: 这里替换真实的vae token embed！
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        # 添加task_embedding
        if hasattr(self, 'task_embedding') and sample_task is not None:
            # 确保sample_task的形状正确
            if sample_task.dim() == 1:
                # 为每个token添加对应的task embedding
                task_embeddings = self.task_embedding(sample_task)
                # sample_task: 当task_id为0时( 对应 t2v task)，不添加embedding
                mask = (sample_task == 0)
                task_embeddings[mask] = 0
                # 将task_embedding添加到packed_sequence
                packed_sequence += task_embeddings

        if hasattr(self, 'modality_embedding') and sample_modality is not None:
            # 确保sample_modality的形状正确
            if sample_modality.dim() == 1:
                # 为每个token添加对应的modality embedding
                modality_embeddings = self.modality_embedding(sample_modality)
                # sample_modality: 当modality_id为0或1时( 对应 text/noise modality)，不添加embedding
                mask = (sample_modality == 0) | (sample_modality == 1)
                modality_embeddings[mask] = 0
                # 将modality_embedding添加到packed_sequence
                packed_sequence += modality_embeddings

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse, frame_mse, total_mse_tokens = None, None, None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            total_mse_tokens = packed_mse_preds.shape[0]
            target = noise - packed_latent_clean  # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            if key_frame_mask != [] and choose_frame_loss:
                mse = (packed_mse_preds - target[has_mse][key_frame_mask==1]) ** 2  # 只对关键帧计算损失
            else:
                mse = (packed_mse_preds - target[has_mse]) ** 2

        # 更鲁棒的实现，忽略了越界的token id
        ce = None
        if ce_loss_indexes is not None:
            V_eff = self.vocab_size_efficient          # 建议等于 len(self.tokenizer)
            ignore_index = -100

            h = last_hidden_state[ce_loss_indexes]      # [L, H]
            logits = self.language_model.lm_head(h)[..., :V_eff]   # [L, V_eff]

            targets = packed_label_ids.to(dtype=torch.long)
            invalid = (targets >= V_eff) | (targets < 0)
            targets = torch.where(invalid, torch.full_like(targets, ignore_index), targets)
            ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=ignore_index)

        return dict(mse=mse, ce=ce, frame_mse=frame_mse, total_mse_tokens=total_mse_tokens)

    @torch.no_grad()
    def validation_gen(
        self,
        val_packed_text_ids: torch.LongTensor,
        val_packed_text_indexes: torch.LongTensor,
        val_packed_vit_tokens: torch.LongTensor,
        val_packed_vit_token_indexes: torch.LongTensor,
        val_sample_lens: List[int],
        val_packed_position_ids: torch.LongTensor,
        val_split_lens: List[int] = None,
        val_attn_modes: List[str] = None,
        val_sample_N_target: List[int] = None,
        vit_video_grid_thw: Optional[torch.IntTensor] = None,  # NOTE: 仅 TI2I 时使用
        vae_video_grid_thw: Optional[torch.IntTensor] = None,
        video_grid_thw: Optional[torch.IntTensor] = None,
        val_mse_loss_indexes: Optional[torch.BoolTensor] = None,
        # for visual generation
        val_packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        val_padded_latent: Optional[torch.Tensor] = None,
        # val_key_frame_mask: Optional[torch.BoolTensor] = None,
        sample_task: Optional[torch.LongTensor] = None,
        sample_modality: Optional[torch.LongTensor] = None,
        video_sizes: List[Tuple[int, int, int]] = [[1, 256, 256]],
        val_padded_videos: torch.Tensor = None,
        timestep_shift: float = 4.0,
        num_timesteps: int = 24,
        # cfg_text
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_vit_scale: float = 1.0, # HACK
        device=None,
        dtype=None,
        new_token_ids=None,
        BLOCK_SIZE: int = 128,
        apply_chat_template: bool = False,
        apply_qwen_2_5_vl_pos_emb: bool = False,
        image_token_id: int = 151655,
        # sample_index: Optional[torch.LongTensor] = None,
        caption: Optional[List[str]] = None,
        index: str = "",
        **kwargs,
    ):

        # 特殊 token
        start_id = new_token_ids["start_of_image"]
        end_id = new_token_ids["end_of_image"]

        pt, ph, pw = self.latent_patch_size

        index_dtype = val_packed_text_ids.dtype

        # --- 为在循环中切片准备累积长度 ---
        cu_sample_lens = torch.nn.functional.pad(torch.cumsum(torch.tensor(val_sample_lens, device=device), dim=0), (1, 0))

        sample_splits = map_splits_to_samples(val_sample_lens, val_split_lens)

        # 每个样本中的 每个 vit token 序列长度
        if val_packed_vit_tokens is not None and vit_video_grid_thw is not None:
            vit_sample_len = vit_video_grid_thw[:, 0] * vit_video_grid_thw[:, 1] * vit_video_grid_thw[:, 2]  # shape: (N,) , N = 1 * 16 * 16,
            cu_vit_sample_lens = torch.cat([torch.zeros(1, device=vit_video_grid_thw.device, dtype=vit_sample_len.dtype), vit_sample_len.cumsum(0)])
            self.vit_model = self.vit_model.to(device=device, dtype=dtype)

            val_packed_vit_tokens = torch.cat(val_packed_vit_tokens, dim=0)

        x_t_all = []
        max_samples = kwargs.get("max_samples", 16)
        num_samples = len(val_sample_lens)
        max_samples = min(num_samples, max_samples)

        gen_idx = 0
        curr_vae_split_idx, curr_vit_split_idx = 0, 0  # curr_vae_split_idx 为生成的图片索引, 在每个样本的N_target=1时与gen_idx相同； curr_vit_split_idx 为 vit split 索引

        padded_videos = []
        # self.logger.info(f"Validation start... (timesteps = {num_timesteps})")
        for i_sample in range(num_samples):
            left, right = sample_splits[i_sample][0], sample_splits[i_sample][-1] + 1
            # --- for interleave ---
            current_split_lens = val_split_lens[left:right]
            current_attn_modes = val_attn_modes[left:right]
            N_noise_element = current_attn_modes.count("noise") + current_attn_modes.count("full_noise") + current_attn_modes.count("full_noise_target")
            N_vit_split = current_attn_modes.count("full")

            if right > len(val_attn_modes):
                break

            # 跳过非 GEN（这里你的逻辑是看有没有 "noise"）
            if N_noise_element<=0:
                curr_vit_split_idx += N_vit_split  # 推进 vit 样本指针
                continue

            # 生成数量控制：用 gen_idx 而不是 i_sample
            if gen_idx >= max_samples:
                break

            # 1. 获取当前样本在整个批次中的切片信息
            sample_start_idx = cu_sample_lens[i_sample]
            sample_end_idx = cu_sample_lens[i_sample + 1]
            current_seq_len = val_sample_lens[i_sample]
            current_pos_ids = val_packed_position_ids[sample_start_idx:sample_end_idx]
            i_sample_task = sample_task[sample_start_idx:sample_end_idx]
            i_sample_modality = sample_modality[sample_start_idx:sample_end_idx]

            # --- 视觉特征 嵌入 ---
            vae_mask = (val_packed_vae_token_indexes >= sample_start_idx) & (val_packed_vae_token_indexes < sample_end_idx)
            current_vae_token_indexes_local = val_packed_vae_token_indexes[vae_mask] - sample_start_idx

            # --- vae mse token 部分 : 指示x_t 中需更新的部分的index ---
            vae_mse_mask = (val_mse_loss_indexes >= sample_start_idx) & (val_mse_loss_indexes < sample_end_idx)
            current_vae_mse_indexes_local = val_mse_loss_indexes[vae_mse_mask] - sample_start_idx  # 指示x_t 中需更新的部分的index
            current_vae_mse_indexes_local_in_vae = (
                current_vae_mse_indexes_local - current_vae_mse_indexes_local[0] + torch.where(current_vae_token_indexes_local == current_vae_mse_indexes_local[0])[0]
            )  # TODO : 如果是多个target image 且中间有文本信息，可能需要修改

            num_vid_tokens_list, vid_shape_list, vae_position_ids, curr_padded_latent = [], [], [], []

            # 2. 其次生成 vit uncond 特征 （可选）
            cfg_vit_pro = False
            if cfg_vit_scale > 1.0 and "full" in current_attn_modes:
                vit_uncond_sequence, vit_uncond_attn_modes, vit_uncond_split_lens, vit_uncond_vae_index, _, vit_uncond_packed_gen_token_indexes, vit_uncond_packed_und_token_indexes, vit_uncond_text_ids, vit_uncond_seq_len, vit_uncond_pad = uncond_split_pro(self.language_model, current_attn_modes, current_split_lens, vae_video_grid_thw, vit_video_grid_thw, curr_vae_split_idx, curr_vit_split_idx, device, dtype, start_id, image_token_id, end_id, BLOCK_SIZE, is_text_uncond = True, is_vit_uncond = True)
                cfg_vit_pro = True

            for i_target in range(N_noise_element):
                T, H, W = video_sizes[curr_vae_split_idx]  # ✅ 尺寸用 gen_idx 索引，保证与“GEN 样本序列顺序”一致
                t = (T - 1) // self.latent_downsample_temporal + 1
                h = H // self.latent_downsample_spatial
                w = W // self.latent_downsample_spatial

                vid_shape_list.append([t, h, w])
                num_vid_tokens_list.append(t * h * w)

                # prepare packed_vae_position_ids
                # 使用3D感知的位置编码函数
                if self.config.interpolate_pos:  # False
                    # 内插
                    vae_position_ids.append(
                        get_flattened_position_ids_interpolate_video(
                            t, h, w, 1, max_num_frames=self.max_num_latent_frames, max_num_patches_per_side=self.max_latent_size  # latent space的patch size为1
                        )
                    )
                else:  # Adopt !!!!
                    # 外插
                    vae_position_ids.append(
                        get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size=self.max_latent_size)  # latent space的patch size为1  # NOT USED in 外插
                    )

                # 当存在condition 的 vae token 时， 计算当前的 padded_latent
                if len(current_vae_mse_indexes_local) != len(current_vae_token_indexes_local):
                    padded_latent_ = val_padded_latent[curr_vae_split_idx]  # (T,H,W,C)

                    patches = rearrange(padded_latent_, "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)", t=t, pt=pt, h=h, ph=ph, w=w, pw=pw)
                    curr_padded_latent.append(patches)

                if val_padded_videos is not None:
                    padded_videos.append(val_padded_videos[curr_vae_split_idx])

                curr_vae_split_idx += 1

            num_vid_tokens = sum(num_vid_tokens_list)
            vae_position_ids = torch.cat(vae_position_ids, 0)
            if curr_padded_latent != []:
                curr_padded_latent = torch.cat(curr_padded_latent, dim=0).to(dtype)

            # 2. 为当前样本重建输入序列和注意力掩码
            current_sequence = torch.zeros((current_seq_len, self.hidden_size), device=device, dtype=dtype)

            # --- 文本部分 ---
            text_mask = (val_packed_text_indexes >= sample_start_idx) & (val_packed_text_indexes < sample_end_idx)
            current_text_indexes_local = val_packed_text_indexes[text_mask] - sample_start_idx

            current_text_ids = val_packed_text_ids[sample_start_idx:sample_end_idx]

            # ++ 如果修改 val_data 和train_data对齐即不使用
            current_text_embedding = self.language_model.model.embed_tokens(current_text_ids).to(dtype=dtype)

            current_sequence[current_text_indexes_local] = current_text_embedding[current_text_indexes_local]

            if cfg_text_scale > 1.0:
                if cfg_vit_pro:
                    vit_uncond_attn_modes_, vit_uncond_split_lens_ = vit_uncond_attn_modes, vit_uncond_split_lens
                    vit_uncond_attn_mask = self.process_attention_mask(vit_uncond_attn_modes_, vit_uncond_split_lens_, [vit_uncond_seq_len, vit_uncond_pad], device = device, BLOCK_SIZE = BLOCK_SIZE)

            # --- vit部分: 支持 ti2i ---
            if N_vit_split != 0:
                vit_sample_start_idx = cu_vit_sample_lens[curr_vit_split_idx]
                vit_sample_end_idx = cu_vit_sample_lens[curr_vit_split_idx + N_vit_split]
                current_val_packed_vit_tokens = val_packed_vit_tokens[vit_sample_start_idx:vit_sample_end_idx].to(dtype)
                current_val_vit_video_grid_thw = vit_video_grid_thw[curr_vit_split_idx : curr_vit_split_idx + N_vit_split]
                curr_vit_split_idx += N_vit_split  # 推进 vit 样本指针

                if self.vit_type in ["qwen2_5_vl", "qwen_2_5_vl_original"]:
                    packed_vit_token_embed = self.vit_model(hidden_states=current_val_packed_vit_tokens, grid_thw=current_val_vit_video_grid_thw)
                    if self.vit_type in ["qwen2_5_vl"]:
                        packed_vit_token_embed = self.connector(packed_vit_token_embed).to(dtype)
                else:
                    raise NotImplementedError(f"{self.vit_type} is not supported")

                vit_mask = (val_packed_vit_token_indexes >= sample_start_idx) & (val_packed_vit_token_indexes < sample_end_idx)
                current_vit_indexes_local = val_packed_vit_token_indexes[vit_mask] - sample_start_idx
                current_sequence[current_vit_indexes_local] = packed_vit_token_embed

            # --- 关键：与训练一致 → pad 到 BLOCK_SIZE 的倍数，并让输入/掩码/length 全一致 ---
            current_seq_len_pad = (current_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            current_pad = current_seq_len_pad - current_seq_len
            if current_pad > 0:
                current_split_lens = current_split_lens + [current_pad]
                current_attn_modes = current_attn_modes + ["causal"]
            current_split_lens_, current_attn_modes_ = current_split_lens, current_attn_modes

            attention_mask = self.process_attention_mask(current_attn_modes_, current_split_lens_,  [current_seq_len, current_pad], device = device, BLOCK_SIZE = BLOCK_SIZE)
            # NOTE: 固定种子
            validation_noise_seed = kwargs.get("validation_noise_seed", -1)
            if validation_noise_seed > 0:
                generator = torch.Generator(device=device).manual_seed(validation_noise_seed + get_global_rank() * max_samples + i_sample)  # 构造seed
            else:
                generator = None
            x_t = torch.randn(num_vid_tokens, self.patch_latent_dim, generator=generator, device=device, dtype=dtype)  # [1*t*h*w, pt*ph*pw*C]

            if curr_padded_latent != []:  # 存在 vae_condition
                curr_padded_latent[current_vae_mse_indexes_local_in_vae] = x_t[current_vae_mse_indexes_local_in_vae]
                x_t = curr_padded_latent

            timesteps = torch.linspace(1, 0, num_timesteps + 1, device=x_t.device)  # fix: 加1
            timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
            dts = timesteps[:-1] - timesteps[1:]
            timesteps = timesteps[:-1]

            if apply_qwen_2_5_vl_pos_emb:
                grid_thw_rope = video_grid_thw[i_sample]

                # Qwen-VL 中求解方法， TODO: rope_deltas在一次性求完所有 token 的 position_ids 时有作用吗
                current_pos_ids, _ = self.language_model.get_rope_index(
                    input_ids=current_text_ids.unsqueeze(0),
                    image_grid_thw=grid_thw_rope, #vae_video_grid_thw[gen_idx : gen_idx + 1],  # NOTE: 这里是全局索引
                    video_grid_thw=grid_thw_rope, #vae_video_grid_thw[gen_idx : gen_idx + 1],  # video_grid_thw,
                    second_per_grid_ts=[1.0]*len(grid_thw_rope),  # second_per_grid_ts,
                    attention_mask=torch.ones([1, len(current_text_ids)], dtype=torch.long, device=device),  # attention_mask, 全1掩码？
                )  # current_packed_position_ids: [3, 1, L]
                # mrope 上区分  ref image vae特征与video vae特征
                current_pos_ids = shift_position_ids(
                    current_pos_ids,
                    pos_shift=1000,
                    attn_modes=current_attn_modes,
                    split_lens=current_split_lens,
                    shift_attn_mode=["full_noise", "full"],
                    pro_type=10,
                    i_sample_task=i_sample_task,
                    i_sample_modality=i_sample_modality,
                )

            if cfg_text_scale > 1.0:
                uncond_mask = i_sample_modality!=0
                _, uncond_pos_ids, uncond_attn_mask, _, _, uncond_extra_inputs, uncond_seq_len = self.uncond_split_pro_new(
                    uncond_mask,
                    current_text_ids,
                    current_attn_modes,
                    current_split_lens,
                    device,
                    dtype,
                    BLOCK_SIZE,
                    grid_thw_rope,
                    apply_qwen_2_5_vl_pos_emb,
                    i_sample_task=i_sample_task,
                    i_sample_modality=i_sample_modality,
                )

            for _ in range(1):
                timestep = torch.zeros(x_t.shape[0], device=x_t.device)
                # for group-by-group generation

                for i, timestep_ in enumerate(timesteps):
                    timestep[current_vae_mse_indexes_local_in_vae] = torch.tensor([timestep_] * current_vae_mse_indexes_local_in_vae.shape[0], device=x_t.device)
                    if timestep_ > cfg_interval[0] and timestep_ <= cfg_interval[1]:
                        cfg_text_scale_ = cfg_text_scale
                        cfg_vit_scale_ = cfg_vit_scale # 默认 vit_uncond 和text_uncond 都采用 同一 cfg_interval
                    else:
                        cfg_text_scale_ = 1.0
                        cfg_vit_scale_ = 1.0

                    # --- 视觉特征 编码 ---
                    timestep_embed = self.time_embedder(timestep)
                    latent_pos_embed = self.latent_pos_embed(vae_position_ids)
                    vae_embed = self.vae2llm(x_t) + timestep_embed + latent_pos_embed
                    vae_embed = vae_embed.to(current_sequence.dtype)

                    # 一切正常再做赋值
                    current_sequence[current_vae_token_indexes_local] = vae_embed

                    extra_inputs = {}  # {'mode': "und"}
                    if self.use_moe: # NOTE: packed_und_token_indexes 的计算可能有问题，在x2v的任务中
                        if N_vit_split != 0:
                            packed_und_token_indexes = torch.cat([current_text_indexes_local, current_vit_indexes_local], dim=0)
                        else:
                            packed_und_token_indexes = current_text_indexes_local
                        extra_inputs.update(
                        packed_und_token_indexes=packed_und_token_indexes.to(dtype=index_dtype),
                        packed_gen_token_indexes=current_vae_token_indexes_local.to(dtype=index_dtype),
                    )

                    self.language_model.to(current_sequence.dtype)
                    cond_hidden_state = self.language_model(
                    packed_sequence=current_sequence[:current_seq_len],  # current_sequence,
                    sample_lens=[current_seq_len],  # [current_seq_len_pad]
                    attention_mask=attention_mask,
                    packed_position_ids=current_pos_ids.to(dtype=index_dtype),
                    mode_forward="validation",
                    **extra_inputs,
                )
                    v_t = self.llm2vae(cond_hidden_state[current_vae_mse_indexes_local])

                    # --- 引入 cfg ---
                    if cfg_text_scale_ > 1.0:
                        uncond_sequence = current_sequence[uncond_mask] # 与uncond_sequence等价
                        cfg_text_v_t = self.uncond_forward(uncond_sequence, uncond_pos_ids, uncond_seq_len, uncond_attn_mask, uncond_extra_inputs, current_vae_mse_indexes_local, current_seq_len)

                        if cfg_vit_pro:
                            if i_sample_task is not None:
                                i_sample_task_text_uncond = i_sample_task[i_sample_modality!=0]
                                i_sample_modality_text_uncond = i_sample_modality[i_sample_modality!=0]
                            else:
                                i_sample_task_text_uncond, i_sample_modality_text_uncond = None, None

                            if i_sample_task is not None:
                                i_sample_task_text_vit_uncond = i_sample_task_text_uncond[i_sample_modality_text_uncond!=4]
                                i_sample_modality_text_vit_uncond = i_sample_modality_text_uncond[i_sample_modality_text_uncond!=4]
                            else:
                                i_sample_task_text_vit_uncond, i_sample_modality_text_vit_uncond = None, None

                            cfg_text_vit_v_t = self.uncond_forward(vae_embed, vit_uncond_sequence, vit_uncond_text_ids, vit_uncond_seq_len, vit_uncond_packed_und_token_indexes, vit_uncond_packed_gen_token_indexes, vit_uncond_attn_mask, vit_uncond_vae_index, grid_thw_rope, current_vae_mse_indexes_local, current_seq_len, apply_qwen_2_5_vl_pos_emb, device,i_sample_task_text_vit_uncond,i_sample_modality_text_vit_uncond)

                            v_t_ = cfg_text_vit_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t) + cfg_vit_scale_  * (cfg_text_v_t - cfg_text_vit_v_t)
                        else:
                            v_t_ = cfg_text_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t)

                        # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                        if cfg_renorm_type == "global":
                            norm_v_t = torch.norm(v_t)
                            norm_v_t_ = torch.norm(v_t_)
                            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        elif cfg_renorm_type == "channel":
                            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                            norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        elif cfg_renorm_type.lower() in ("", "none", "null"):
                            scale = 1
                        else:
                            raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                        v_t = v_t_ * scale

                    x_t[current_vae_mse_indexes_local_in_vae] = x_t[current_vae_mse_indexes_local_in_vae] - v_t.to(x_t.device) * dts[i]  # velocity pointing from data to noise

            # ---- 每个样本各自重排到 [T,H,W,C]，避免用最后一个样本的 t/h/w 去重排整批 ----
            curr_seq_target, patch = 0, []
            for i_target in range(N_noise_element):

                pt, ph, pw = self.latent_patch_size
                t, h, w = vid_shape_list[i_target]
                len_target = t * h * w

                x_t_ =  rearrange(x_t[curr_seq_target : curr_seq_target + len_target], "(t h w) (pt ph pw c) -> (t pt) (h ph) (w pw) c", t=t, h=h, w=w, pt=pt, ph=ph, pw=pw)

                patch.append(x_t_)
                curr_seq_target += len_target
            # patch = torch.cat(patch, dim=0)
            x_t_all.append(patch)
            # ✅ 成功处理一个 GEN 样本，推进 gen_idx
            gen_idx += 1

        # self.logger.info(f"Validation step done with {len(x_t_all)} samples.")

        if caption != None:
            return x_t_all, [caption], padded_videos, index

        # if padded_videos != []:
        #     return x_t_all, padded_videos

        return x_t_all

    def uncond_split_pro_new(
        self,
        uncond_mask,
        current_text_ids,
        current_attn_modes,
        current_split_lens,
        device,
        dtype,
        BLOCK_SIZE,
        grid_thw_rope=None,
        apply_qwen_2_5_vl_pos_emb=False,
        i_sample_task=None,
        i_sample_modality=None,
        uncond_pos_ids=None,
    ):
        """
        uncond_attn_modes: uncond 序列的 attn_mode 列表
        uncond_split_lens: uncond 序列的 split_len 列表
        uncond_packed_gen_token_indexes: uncond 序列中 gen_token 索引列表 (for extra_inputs)
        uncond_packed_und_token_indexes: uncond 序列中 und_token 索引列表 (for extra_inputs)
        uncond_seq_len: uncond 序列的长度
        uncond_pad: uncond 序列的 pad 长度
        """
        start = 0
        uncond_split_lens, uncond_attn_modes, uncond_packed_gen_token_indexes = [], [], []
        for i_visual, attn_mode_ in enumerate(current_attn_modes):
            split_len_ = current_split_lens[i_visual]
            end = start + split_len_
            split_in_uncond = int(uncond_mask[start:end].sum())
            start += split_len_
            if split_in_uncond == 0:  # 即表示当前split在uncond 序列中不保留
                continue
            else:
                if attn_mode_ in ["noise", "full_noise"]:
                    start_gen, end_gen = sum(uncond_split_lens) + 1, sum(uncond_split_lens) + 1 + split_len_ - 2
                    uncond_packed_gen_token_indexes.extend(range(start_gen, end_gen))
                uncond_split_lens.append(split_in_uncond)  # 保留对应长度，一般除非使用system_prompt，否则split_in_uncond == split_len_
                uncond_attn_modes.append(attn_mode_)

        # 与训练一致 → 也 pad 掉尾块
        uncond_seq_len = sum(uncond_split_lens)
        uncond_seq_len_pad = (uncond_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        uncond_pad = uncond_seq_len_pad - uncond_seq_len
        if uncond_pad > 0:
            uncond_split_lens.append(uncond_pad)
            uncond_attn_modes.append("causal")

        uncond_packed_gen_token_indexes = torch.tensor(uncond_packed_gen_token_indexes, dtype=torch.long, device=device)
        all_indexes = torch.arange(0, uncond_seq_len).to(device)
        und_token_mask = ~torch.isin(all_indexes, uncond_packed_gen_token_indexes)  # 判断 A 中哪些元素不在 B 中
        uncond_packed_und_token_indexes = all_indexes[und_token_mask]

        uncond_extra_inputs = {}  # {'mode': "validation"}
        if self.use_moe:
            uncond_extra_inputs.update(
                packed_und_token_indexes=uncond_packed_und_token_indexes,
                packed_gen_token_indexes=uncond_packed_gen_token_indexes,
            )

        # uncond attention mask 的处理
        uncond_attn_mask = self.process_attention_mask(uncond_attn_modes, uncond_split_lens, [uncond_seq_len, uncond_pad], device=device, BLOCK_SIZE=BLOCK_SIZE)

        # 提取 uncond 序列的 text_ids
        uncond_text_ids = current_text_ids[uncond_mask]
        uncond_sample_task = i_sample_task[uncond_mask] if i_sample_task is not None else None
        uncond_sample_modality = i_sample_modality[uncond_mask] if i_sample_modality is not None else None

        if apply_qwen_2_5_vl_pos_emb:
            uncond_pos_ids, uncond_rope_deltas = self.language_model.get_rope_index(
                input_ids=uncond_text_ids.unsqueeze(0),
                image_grid_thw=grid_thw_rope,  # vae_video_grid_thw[gen_idx : gen_idx + 1],
                video_grid_thw=grid_thw_rope,  # vae_video_grid_thw[gen_idx : gen_idx + 1],  # video_grid_thw,
                second_per_grid_ts=[1.0] * len(grid_thw_rope),  # second_per_grid_ts,
                attention_mask=torch.ones([1, len(uncond_text_ids)], dtype=torch.long, device=device),  # attention_mask, 全1掩码？
            )
            # mrope 上区分  ref image vae特征与video vae特征
            uncond_pos_ids = shift_position_ids(
                uncond_pos_ids,
                pos_shift=1000,
                attn_modes=uncond_attn_modes,
                split_lens=uncond_split_lens,
                shift_attn_mode=["full_noise", "full"],
                pro_type=10,
                i_sample_task=uncond_sample_task,
                i_sample_modality=uncond_sample_modality,
            )
        else:
            uncond_pos_ids = torch.tensor(uncond_pos_ids, dtype=torch.long, device=device)[:uncond_seq_len]

        return (
            uncond_text_ids,
            uncond_pos_ids,
            uncond_attn_mask,
            uncond_attn_modes,
            uncond_split_lens,
            uncond_extra_inputs,
            uncond_seq_len,
        )

    def uncond_forward(
        self,
        uncond_sequence,
        uncond_pos_ids,
        uncond_seq_len,
        uncond_attn_mask,
        uncond_extra_inputs,
        current_vae_mse_indexes_local,
        current_seq_len,
    ):
        # 执行无条件前向传播
        uncond_hidden_state = self.language_model(
            packed_sequence=uncond_sequence[:uncond_seq_len],
            sample_lens=[uncond_seq_len],
            attention_mask=uncond_attn_mask,
            packed_position_ids=uncond_pos_ids,
            mode_forward="validation",  # NOTE
            **uncond_extra_inputs,
        )
        uncond_current_vae_mse_indexes_local = current_vae_mse_indexes_local - (current_seq_len - uncond_seq_len)  # TODO : 如果是多个target image 且中间有文本信息，可能需要修改
        cfg_text_v_t = self.llm2vae(uncond_hidden_state[uncond_current_vae_mse_indexes_local])

        return cfg_text_v_t

    @torch.no_grad()
    def validation_video_to_text(
        self,
        val_packed_text_ids: torch.LongTensor,
        val_packed_text_indexes: torch.LongTensor,
        val_packed_position_ids: torch.LongTensor,
        val_ce_loss_indexes: torch.LongTensor,
        val_sample_N_target: List[int],  # 理论上 und 分支的 N_target 均为 1
        val_split_lens: List[int],
        val_attn_modes: List[str],
        val_sample_lens: List[int],
        val_sample_type: List[str],
        # val_split_lens: List[int] = None,
        # val_attn_modes: List[str] = None,
        val_packed_vit_tokens: Optional[torch.Tensor] = None,
        # val_packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        # val_packed_vit_position_ids: Optional[torch.LongTensor] = None,
        # val_vit_token_seqlens: Optional[torch.IntTensor] = None,
        val_vit_video_grid_thw: Optional[torch.IntTensor] = None,  # for video understanding
        max_samples: int = 1,
        max_length: int = 256,
        device: torch.device = None,
        dtype: torch.dtype = None,
        new_token_ids: Dict[str, int] = None,
        pad_token_id: int = None,
        vocab_size: int = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        caption: any = "",
        tokenizer: any = None,  # 适应有instruction的处理
        apply_chat_template: bool = False,
        apply_qwen_2_5_vl_pos_emb: bool = False,
        image_token_id: int = 151655,
        BLOCK_SIZE: int = 128,
        visualize_generation_progress: bool = False,
        index: str = "",
    ):
        # 特殊 token
        start_id = new_token_ids["start_of_image"]
        end_id = new_token_ids["end_of_image"]
        bos_id = new_token_ids["bos_token_id"]
        eos_id = new_token_ids["eos_token_id"]

        # 每个样本长度
        cu_sample_lens = torch.nn.functional.pad(torch.cumsum(torch.tensor(val_sample_lens, device=device), dim=0), (1, 0))
        sample_splits = map_splits_to_samples(val_sample_lens, val_split_lens)

        # 每个样本中的 每个 vit token 序列长度
        vit_sample_len = val_vit_video_grid_thw[:, 0] * val_vit_video_grid_thw[:, 1] * val_vit_video_grid_thw[:, 2]  # shape: (N,) , N = 1 * 16 * 16,
        cu_vit_sample_lens = torch.cat([torch.zeros(1, device=val_vit_video_grid_thw.device, dtype=vit_sample_len.dtype), vit_sample_len.cumsum(0)])

        if val_packed_vit_tokens is not None:
            val_packed_vit_tokens = torch.cat(val_packed_vit_tokens, dim=0)

        # max_samples = min(len(val_sample_lens) - 1, max_samples)
        max_samples = min(len(val_sample_lens), max_samples)  #  NOTE 与测试的时候兼容，不再-1
        cnt_samples = 0
        generated_sequence_all = []

        # L = len(val_sample_lens) - 1
        L = len(val_sample_lens)  # 与测试的时候兼容，不再-1
        curr_vit_split_idx = 0
        for i_sample in range(L):
            left, right = sample_splits[i_sample][0], sample_splits[i_sample][-1] + 1
            # --- for interleave ---
            current_split_lens = val_split_lens[left:right]
            current_attn_modes = val_attn_modes[left:right]
            N_target = val_sample_N_target[i_sample]  # 理论上 und 分支的 N_target 均为 1
            N_vit_split = current_attn_modes.count("full")

            if val_sample_type[i_sample] != "und":
                curr_vit_split_idx += N_vit_split  # 推进 vit 样本指针
                continue
            cnt_samples += 1
            if cnt_samples > max_samples:
                break

            assert N_target == 1

            # 获取当前video vit样本在整个批次中的切片信息
            vit_sample_start_idx = cu_vit_sample_lens[curr_vit_split_idx]
            vit_sample_end_idx = cu_vit_sample_lens[curr_vit_split_idx + N_vit_split]
            current_val_packed_vit_tokens = val_packed_vit_tokens[vit_sample_start_idx:vit_sample_end_idx]
            current_val_vit_video_grid_thw = val_vit_video_grid_thw[curr_vit_split_idx : curr_vit_split_idx + N_vit_split]
            curr_vit_split_idx += N_vit_split  # 推进 vit 样本指针

            if N_vit_split > 0 :
                if self.vit_type in ["qwen2_5_vl", "qwen_2_5_vl_original"]:
                    packed_vit_token_embed = self.vit_model(hidden_states=current_val_packed_vit_tokens, grid_thw=current_val_vit_video_grid_thw)
                    if self.vit_type in ["qwen2_5_vl"]:
                        packed_vit_token_embed = self.connector(packed_vit_token_embed).to(dtype)
                else:
                    raise NotImplementedError(f"{self.vit_type} is not supported")

            # 获取当前 文本条件 （包括特殊token） 在整个批次中的切片信息
            sample_start_idx = cu_sample_lens[i_sample]
            sample_end_idx = cu_sample_lens[i_sample + 1]
            current_pos_ids = val_packed_position_ids[sample_start_idx:sample_end_idx]

            # text_mask = (val_packed_text_indexes >= sample_start_idx) & (val_packed_text_indexes < sample_end_idx)
            # current_text_indexes_local = val_packed_text_indexes[text_mask] - sample_start_idx

            text_mask_ce = (val_ce_loss_indexes >= sample_start_idx) & (val_ce_loss_indexes < sample_end_idx)
            current_ce_loss_indexes_local = val_ce_loss_indexes[text_mask_ce] - sample_start_idx
            if text_mask_ce.numel() != 0:
                current_text_ids = val_packed_text_ids[sample_start_idx:sample_end_idx][: current_ce_loss_indexes_local[0] + 1]
            else:
                current_text_ids = val_packed_text_ids[sample_start_idx:sample_end_idx]

            num_text_ids = current_text_ids.shape[0]
            num_last_split = num_text_ids - sum(current_split_lens[:-N_target])

            current_split_lens = current_split_lens[:-N_target]  # 去除目标序列的split 长度

            if num_last_split > 1:
                current_split_lens.extend([num_last_split - 1])  # num_last_split 包含了 起始token ， 所以需要减1

            # NOTE 填充pad，让总的seqlen能够被BLOCK_SIZE整除！
            max_seq_len = (max_length + num_text_ids + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            num_pad = max_seq_len - num_text_ids

            current_text_ids = torch.cat(
                [current_text_ids, torch.full((num_pad,), pad_token_id, dtype=torch.long, device=device)], dim=0
            )  # [151652, 151653, 151654] + [151643] * 61,  151643 = <|endoftext|>
            packed_text_embedding = self.language_model.model.embed_tokens(current_text_ids).to(dtype)  # [64, 1536]

            # 把packed_vit_token_embed插入image_pad对应位置
            if N_vit_split > 0 :
                mask = current_text_ids == image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(packed_text_embedding)
                image_mask = mask_expanded.to(packed_text_embedding.device)
                curr_packed_sequence = packed_text_embedding.masked_scatter(image_mask, packed_vit_token_embed)
            else:
                curr_packed_sequence = packed_text_embedding

            # begin text generation
            step = num_text_ids - 1  # NOTE step会指示当前生成到第几个token，一开始的时候，应该定位到BOS / 起始token (比如"assistant\n"之后)
            generated_sequence = []

            if apply_qwen_2_5_vl_pos_emb:
                # Qwen-VL 中求解方法， TODO: rope_deltas在一次性求完所有 token 的 position_ids 时有作用吗
                current_packed_position_ids, rope_deltas = self.language_model.get_rope_index(
                    input_ids=current_text_ids.unsqueeze(0),
                    image_grid_thw=current_val_vit_video_grid_thw,
                    video_grid_thw=current_val_vit_video_grid_thw,  # video_grid_thw,
                    second_per_grid_ts=[1.0],  # second_per_grid_ts,
                    attention_mask=torch.ones([1, max_seq_len], dtype=torch.long, device=device),  # attention_mask, 全1掩码？
                )  # current_packed_position_ids: [3, 1, L]
            else:
                current_pos_ids = current_pos_ids[:num_text_ids]
                pos_pad_start = int(current_pos_ids[-1] + 1)
                current_pad = torch.arange(pos_pad_start, pos_pad_start + num_pad, device=device)
                current_packed_position_ids = torch.cat([current_pos_ids, current_pad], dim=0)

            current_sample_lens = [max_seq_len]
            seqlen = sum(current_sample_lens)  # 始终是max_length
            current_attn_modes_ = current_attn_modes[: len(current_split_lens)] + ["causal", "causal"]
            current_attn_modes_ = ["full" if mode_=="full_noise" else mode_ for mode_ in current_attn_modes_]
            while step < (max_seq_len - 1):
                # current_sample_lens = [curr_packed_sequence.shape[0]]
                # attention 的求解存在问题

                current_text_len = (step + 1) - (num_text_ids - 1)
                current_split_lens_ = current_split_lens + [current_text_len, num_pad + 1 - current_text_len]

                sparse_mask = create_sparse_mask(current_sample_lens, current_split_lens_, current_attn_modes_, device)
                attention_mask = create_block_mask(sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, device=device, BLOCK_SIZE=BLOCK_SIZE, _compile=False)

                # 构建MoE输入
                extra_inputs = {"mode": "und"}
                if self.use_moe:
                    packed_und_token_indexes = torch.arange(0, max_seq_len, device=device)  # all tokens are UND
                    extra_inputs.update(
                        packed_und_token_indexes=packed_und_token_indexes,
                        packed_gen_token_indexes=None,
                    )

                last_hidden_state = self.language_model(
                    packed_sequence=curr_packed_sequence.to(dtype=dtype),
                    sample_lens=current_sample_lens,
                    attention_mask=attention_mask,
                    packed_position_ids=current_packed_position_ids,  # 这里正确嘛？
                    mode_forward="validation",
                    **extra_inputs,
                )

                # sample
                pred_logits = self.language_model.lm_head(last_hidden_state[step : step + 1, :])
                # 将大于vocab_size的pred_logits设置成负无穷
                pred_logits[:, vocab_size:] = float("-inf")
                if do_sample:
                    probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                    curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    curr_tokens = torch.argmax(pred_logits, dim=-1)

                generated_sequence.append(curr_tokens)
                if visualize_generation_progress:
                    print(f"curr_tokens: {curr_tokens}", curr_tokens.item(), ", eos_id:", eos_id)

                if curr_tokens.item() == eos_id:
                    break

                # curr_tokens替代当前step的pad embeddings
                curr_packed_sequence[step + 1] = self.language_model.model.embed_tokens(curr_tokens)

                step += 1

            generated_sequence = torch.stack([i.to(device) for i in generated_sequence], dim=0)  # [L, 1] , torch.int64
            generated_sequence_all.append(generated_sequence)
        return generated_sequence_all, caption, index

    def validation_video_to_text_KVcache(
        self,
        val_attn_modes: List[str],
        val_sample_lens: List[int],
        val_packed_vit_tokens: Optional[torch.Tensor] = None,
        val_vit_video_grid_thw: Optional[torch.IntTensor] = None,  # for video understanding
        max_samples: int = 1,
        max_length: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = None,
        new_token_ids: Dict[str, int] = None,
        pad_token_id: int = None,
        vocab_size: int = None,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        # 每个样本的 token 数
        sample_len = (
            val_vit_video_grid_thw[:, 0]
            * val_vit_video_grid_thw[:, 1]
            * val_vit_video_grid_thw[:, 2]
        )
        # ✅ 前缀和长度做成 N+1：[0, s1, s1+s2, ...]
        cu_sample_lens = torch.cat(
            [
                torch.zeros(1, device=val_vit_video_grid_thw.device, dtype=sample_len.dtype),
                sample_len.cumsum(0),
            ],
            dim=0,
        )
        # self.logger.info(f'cu_sample_lens: {cu_sample_lens}')

        max_samples = min(len(val_sample_lens) - 1, max_samples)
        generated_sequence_all = []

        cnt_samples = 0
        L = len(val_sample_lens) - 1
        und_idx = 0  # ✅ 只在遇到 UND 样本时递增，用它索引 UND 专属的张量/前缀和

        # self.logger.info(f'val_attn_modes: {val_attn_modes}')

        for i_sample in range(L):
            left, right = 2 * i_sample, 2 * i_sample + 1
            # self.logger.info(f'left: {left}, right: {right}')
            if right >= len(val_attn_modes):   # ✅ 边界判断
                break

            # GEN: ('causal', 'noise') -> 跳过；UND 留下
            if not (val_attn_modes[left] == "full" and val_attn_modes[right] == "causal"):
                # 不是und模式，跳过
                continue

            # ✅ 用 UND 的真实下标切片
            if und_idx + 1 >= cu_sample_lens.numel():
                break  # 安全兜底，防止坏数据越界

            cnt_samples += 1
            if cnt_samples > max_samples:
                break

            sample_start_idx = cu_sample_lens[und_idx].item()
            sample_end_idx   = cu_sample_lens[und_idx + 1].item() # fix: 用 und_idx 来索引，而不是 i_sample

            current_val_packed_vit_tokens = val_packed_vit_tokens[sample_start_idx:sample_end_idx]
            current_val_vit_video_grid_thw = val_vit_video_grid_thw[und_idx:und_idx + 1]
            und_idx += 1  # ✅ 遇到一个 UND，推进一次

            # 单样本逐一处理时使用，同时处理多个样本需修改
            past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
            generation_input, kv_lens, ropes = self.prepare_vit_images_validation(curr_kvlens=[0], curr_rope=[0], vit_tokens = [current_val_packed_vit_tokens], new_token_ids=new_token_ids, device=device)
            past_key_values = self.forward_cache_update_vit_validation(past_key_values, vit_vae_video_grid_thw=current_val_vit_video_grid_thw, device=device, dtype=dtype,**generation_input)

            # initial start token
            generation_input = self.prepare_start_tokens(kv_lens, ropes, new_token_ids, device=device)
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids["eos_token_id"],
                vocab_size=vocab_size,
                **generation_input,
            )  # [L,1]
            generated_sequence_all.append(unpacked_latent)
        return generated_sequence_all

    # 单样本情况下: curr_kvlens :  [0], curr_rope: [0], images: <PIL.Image.Image image mode=RGB>
    def prepare_vit_images_validation(self, curr_kvlens, curr_rope, vit_tokens, new_token_ids, device):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for vit_token, curr_kvlen, curr_position_id in zip(vit_tokens, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_vit_tokens.append(vit_token)
            num_img_tokens = len(vit_tokens[0]) // 4  # 实际上qwen2.5-vl还需要merge，2x2 merge成1个，
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad()
    def forward_cache_update_vit_validation(
        self,
        past_key_values: NaiveCache,
        vit_vae_video_grid_thw: torch.IntTensor, ## ++
        packed_text_ids: torch.LongTensor, ## 是否包含special token
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids).to(dtype)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size), dtype = dtype)
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if self.vit_type in ["qwen2_5_vl", "qwen_2_5_vl_original"]:
            # NOTE video understanding部分
            packed_vit_token_embed = self.vit_model(
                hidden_states=packed_vit_tokens,
                grid_thw=vit_vae_video_grid_thw,
            )
            if self.vit_type in ["qwen2_5_vl"]:
                packed_vit_token_embed = self.connector(packed_vit_token_embed).to(dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
        else:
            raise NotImplementedError(f"{self.vit_type} is not supported")

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values


    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids, device):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids["bos_token_id"])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long).to(device),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long).to(device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int).to(device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long).to(device),
        }

        return generation_input

    @torch.no_grad()
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
        vocab_size: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            pred_logits[:, vocab_size:] = float('-inf') # ++
            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat([uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0)
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0].item() == end_token_id:  # only support batch=1
                generated_sequence.append(curr_tokens)
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)



    # ------------------------------------------------------------------------------------------------------------------------------
    # for validation_gen_kvcache
    def init_gen_context(self, device: torch.device, dtype: torch.dtype):
        gen_context = {
            'kv_lens': torch.tensor([0], device=device, dtype=dtype), #kv_lens 代表 "key-value lengths"，用于追踪注意力机制（Attention Mechanism）中已经缓存的键（key）和值（value）的序列长度。初始化为 [0] 表示在最开始，这个“记忆”是空的，还没有处理任何输入。
            #'ropes': [0], #旋转位置编码（Rotary Position Embeddings, RoPE）的状态，RoPE 是一种给模型输入序列提供位置信息的方法。这里初始化为 [0] 同样表示从序列的第一个位置开始。
            'past_key_values': NaiveCache(self.config.llm_config.num_hidden_layers), #用于存储过去所有时间步计算出的注意力键（key）和值（value）。在生成下一个 token 时，模型可以直接使用这些缓存好的 past_key_values，而无需重新计算整个输入序列的注意力，从而极大地提升了生成速度。
        }
        return gen_context


    @torch.no_grad()
    def validation_gen_KVcache(
        self,
        val_packed_text_ids: torch.LongTensor,
        val_packed_text_indexes: torch.LongTensor,
        val_packed_vit_tokens: torch.LongTensor,
        val_packed_vit_token_indexes: torch.LongTensor,
        val_sample_lens: List[int],
        val_packed_position_ids: torch.LongTensor,
        val_split_lens: List[int] = None,
        val_attn_modes: List[str] = None,
        val_sample_N_target: List[int] = None,
        vit_video_grid_thw: Optional[torch.IntTensor] = None,  # NOTE: 仅 TI2I 时使用
        vae_video_grid_thw: Optional[torch.IntTensor] = None,
        video_grid_thw: Optional[torch.IntTensor] = None,
        val_mse_loss_indexes: Optional[torch.BoolTensor] = None,
        # for visual generation
        val_packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        val_padded_latent: Optional[torch.Tensor] = None,
        # val_key_frame_mask: Optional[torch.BoolTensor] = None,
        sample_task: Optional[torch.LongTensor] = None,
        sample_modality: Optional[torch.LongTensor] = None,
        video_sizes: List[Tuple[int, int, int]] = [[1, 256, 256]],
        val_padded_videos: torch.Tensor = None,
        timestep_shift: float = 4.0,
        num_timesteps: int = 24,
        # cfg_text
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_vit_scale: float = 1.0, # NOTE ： 对应 cfg_vision_scale
        device=None,
        dtype=None,
        new_token_ids=None,
        BLOCK_SIZE: int = 128,
        apply_chat_template: bool = False,
        apply_qwen_2_5_vl_pos_emb: bool = False,
        image_token_id: int = 151655,
        # sample_index: Optional[torch.LongTensor] = None,
        caption: Optional[List[str]] = None,
        index: str = "",
        **kwargs,
    ):
        cfg_vision_scale = cfg_vit_scale

        pt, ph, pw = self.latent_patch_size

        index_dtype = val_packed_text_ids.dtype

        # --- 为在循环中切片准备累积长度 ---
        cu_sample_lens = torch.nn.functional.pad(torch.cumsum(torch.tensor(val_sample_lens, device=device), dim=0), (1, 0))

        sample_splits = map_splits_to_samples(val_sample_lens, val_split_lens)

        # 每个样本中的 每个 vit token 序列长度
        if val_packed_vit_tokens is not None and vit_video_grid_thw is not None:
            vit_sample_len = vit_video_grid_thw[:, 0] * vit_video_grid_thw[:, 1] * vit_video_grid_thw[:, 2]  # shape: (N,) , N = 1 * 16 * 16,
            cu_vit_sample_lens = torch.cat([torch.zeros(1, device=vit_video_grid_thw.device, dtype=vit_sample_len.dtype), vit_sample_len.cumsum(0)])
            self.vit_model = self.vit_model.to(device=device, dtype=dtype)

            val_packed_vit_tokens = torch.cat(val_packed_vit_tokens, dim=0)

        x_t_all = []
        max_samples = kwargs.get("max_samples", 16)
        L = max(len(val_sample_lens) - 1, 1)
        max_samples = min(L, max_samples)  # update

        gen_idx = 0
        curr_vae_split_idx, curr_vit_split_idx = 0, 0  # curr_vae_split_idx 为生成的图片索引, 在每个样本的N_target=1时与gen_idx相同； curr_vit_split_idx 为 vit split 索引

        padded_videos = []
        # self.logger.info(f"Validation start... (timesteps = {num_timesteps})")
        for i_sample in range(L):  # fix: 需要-1
            left, right = sample_splits[i_sample][0], sample_splits[i_sample][-1] + 1
            # --- for interleave ---
            current_split_lens = val_split_lens[left:right]
            current_attn_modes = val_attn_modes[left:right]
            N_target = val_sample_N_target[i_sample]  # 判断目标图像序列数量
            N_noise_element = current_attn_modes.count("noise") + current_attn_modes.count("full_noise") + current_attn_modes.count("full_noise_target")
            N_vit_split = current_attn_modes.count("full")

            if right > len(val_attn_modes):
                break

            # 跳过非 GEN（这里你的逻辑是看有没有 "noise"）
            if N_noise_element<=0:
                curr_vit_split_idx += N_vit_split  # 推进 vit 样本指针
                continue

            # 生成数量控制：用 gen_idx 而不是 i_sample
            if gen_idx >= max_samples:
                break

            # 1. 获取当前样本在整个批次中的切片信息
            sample_start_idx = cu_sample_lens[i_sample]
            sample_end_idx = cu_sample_lens[i_sample + 1]
            current_seq_len = val_sample_lens[i_sample]
            current_pos_ids = val_packed_position_ids[sample_start_idx:sample_end_idx]
            i_sample_task = sample_task[sample_start_idx:sample_end_idx]
            i_sample_modality = sample_modality[sample_start_idx:sample_end_idx]

            # --- 视觉特征 嵌入 ---
            vae_mask = (val_packed_vae_token_indexes >= sample_start_idx) & (val_packed_vae_token_indexes < sample_end_idx)
            current_vae_token_indexes_local = val_packed_vae_token_indexes[vae_mask] - sample_start_idx

            # --- vae mse token 部分 : 指示x_t 中需更新的部分的index ---
            vae_mse_mask = (val_mse_loss_indexes >= sample_start_idx) & (val_mse_loss_indexes < sample_end_idx)
            current_vae_mse_indexes_local = val_mse_loss_indexes[vae_mse_mask] - sample_start_idx  # 指示x_t 中需更新的部分的index
            current_vae_mse_indexes_local_in_vae = (
                current_vae_mse_indexes_local - current_vae_mse_indexes_local[0] + torch.where(current_vae_token_indexes_local == current_vae_mse_indexes_local[0])[0]
            )  # TODO : 如果是多个target image 且中间有文本信息，可能需要修改

            num_vid_tokens_list, vid_shape_list, vae_position_ids, curr_padded_latent = [], [], [], []

            # 2. 其次生成 vit uncond 特征 （可选）
            cfg_vision_pro = False
            if cfg_vision_scale > 1.0 and "full" in current_attn_modes:
                cfg_vision_pro = True
                vision_uncond_mask =  i_sample_modality <= 1 # i_sample_modality!=4  则为 cfg_vit
                _, vision_uncond_pos_ids, _ = self.uncond_split_pro_kvcache(vision_uncond_mask, current_text_ids, device, dtype, apply_qwen_2_5_vl_pos_emb, grid_thw_rope = grid_thw_rope[-N_target:], current_attn_modes=current_attn_modes, current_split_lens=current_split_lens, i_sample_task=i_sample_task, i_sample_modality=i_sample_modality ) # NOTE: grid_thw_rope 去掉 vit/vae condition 的项

            for i_target in range(N_noise_element):
                T, H, W = video_sizes[curr_vae_split_idx]  # ✅ 尺寸用 gen_idx 索引，保证与“GEN 样本序列顺序”一致
                t = (T - 1) // self.latent_downsample_temporal + 1
                h = H // self.latent_downsample_spatial
                w = W // self.latent_downsample_spatial

                vid_shape_list.append([t, h, w])
                num_vid_tokens_list.append(t * h * w)

                # prepare packed_vae_position_ids
                # 使用3D感知的位置编码函数
                if self.config.interpolate_pos:  # False
                    # 内插
                    vae_position_ids.append(
                        get_flattened_position_ids_interpolate_video(
                            t, h, w, 1, max_num_frames=self.max_num_latent_frames, max_num_patches_per_side=self.max_latent_size  # latent space的patch size为1
                        )
                    )
                else:  # Adopt !!!!
                    # 外插
                    vae_position_ids.append(
                        get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size=self.max_latent_size)  # latent space的patch size为1  # NOT USED in 外插
                    )

                # 当存在condition 的 vae token 时， 计算当前的 padded_latent
                if len(current_vae_mse_indexes_local) != len(current_vae_token_indexes_local):
                    padded_latent_ = val_padded_latent[curr_vae_split_idx]  # (T,H,W,C)

                    patches = rearrange(padded_latent_, "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)", t=t, pt=pt, h=h, ph=ph, w=w, pw=pw)
                    curr_padded_latent.append(patches)

                if val_padded_videos is not None:
                    padded_videos.append(val_padded_videos[curr_vae_split_idx])

                curr_vae_split_idx += 1

            num_vid_tokens = sum(num_vid_tokens_list)
            vae_position_ids = torch.cat(vae_position_ids, 0)
            if curr_padded_latent != []:
                curr_padded_latent = torch.cat(curr_padded_latent, dim=0).to(dtype)

            # 2. 为当前样本重建输入序列和注意力掩码
            current_sequence = torch.zeros((current_seq_len, self.hidden_size), device=device, dtype=dtype)

            # --- 文本部分 ---
            text_mask = (val_packed_text_indexes >= sample_start_idx) & (val_packed_text_indexes < sample_end_idx)
            current_text_indexes_local = val_packed_text_indexes[text_mask] - sample_start_idx

            current_text_ids = val_packed_text_ids[sample_start_idx:sample_end_idx]

            # ++ 如果修改 val_data 和train_data对齐即不使用
            current_text_embedding = self.language_model.model.embed_tokens(current_text_ids).to(dtype=dtype)

            current_sequence[current_text_indexes_local] = current_text_embedding[current_text_indexes_local]

            # --- vit部分: 支持 ti2i ---
            if N_vit_split != 0:
                vit_sample_start_idx = cu_vit_sample_lens[curr_vit_split_idx]
                vit_sample_end_idx = cu_vit_sample_lens[curr_vit_split_idx + N_vit_split]
                current_val_packed_vit_tokens = val_packed_vit_tokens[vit_sample_start_idx:vit_sample_end_idx].to(dtype)
                current_val_vit_video_grid_thw = vit_video_grid_thw[curr_vit_split_idx : curr_vit_split_idx + N_vit_split]
                curr_vit_split_idx += N_vit_split  # 推进 vit 样本指针

                if self.vit_type in ["qwen2_5_vl", "qwen_2_5_vl_original"]:
                    packed_vit_token_embed = self.vit_model(hidden_states=current_val_packed_vit_tokens, grid_thw=current_val_vit_video_grid_thw)
                    if self.vit_type in ["qwen2_5_vl"]:
                        packed_vit_token_embed = self.connector(packed_vit_token_embed).to(dtype)
                else:
                    raise NotImplementedError(f"{self.vit_type} is not supported")

                vit_mask = (val_packed_vit_token_indexes >= sample_start_idx) & (val_packed_vit_token_indexes < sample_end_idx)
                current_vit_indexes_local = val_packed_vit_token_indexes[vit_mask] - sample_start_idx
                current_sequence[current_vit_indexes_local] = packed_vit_token_embed

            # --- 关键：与训练一致 → pad 到 BLOCK_SIZE 的倍数，并让输入/掩码/length 全一致 ---
            current_seq_len_pad = (current_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
            current_pad = current_seq_len_pad - current_seq_len
            if current_pad > 0:
                current_split_lens = current_split_lens + [current_pad]
                current_attn_modes = current_attn_modes + ["causal"]

            # NOTE: 固定种子
            validation_noise_seed = kwargs.get("validation_noise_seed", -1)
            if validation_noise_seed > 0:
                generator = torch.Generator(device=device).manual_seed(validation_noise_seed + get_global_rank() * max_samples + i_sample)  # 构造seed
            else:
                generator = None
            x_t = torch.randn(num_vid_tokens, self.patch_latent_dim, generator=generator, device=device, dtype=dtype)  # [1*t*h*w, pt*ph*pw*C]

            if curr_padded_latent != []:  # 存在 vae_condition
                curr_padded_latent[current_vae_mse_indexes_local_in_vae] = x_t[current_vae_mse_indexes_local_in_vae]
                x_t = curr_padded_latent

            timesteps = torch.linspace(1, 0, num_timesteps + 1, device=x_t.device)  # fix: 加1
            timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
            dts = timesteps[:-1] - timesteps[1:]
            timesteps = timesteps[:-1]

            if apply_qwen_2_5_vl_pos_emb:
                grid_thw_rope = video_grid_thw[i_sample]

                # Qwen-VL 中求解方法， TODO: rope_deltas在一次性求完所有 token 的 position_ids 时有作用吗
                current_pos_ids, _ = self.language_model.get_rope_index(
                    input_ids=current_text_ids.unsqueeze(0),
                    image_grid_thw=grid_thw_rope, #vae_video_grid_thw[gen_idx : gen_idx + 1],  # NOTE: 这里是全局索引
                    video_grid_thw=grid_thw_rope, #vae_video_grid_thw[gen_idx : gen_idx + 1],  # video_grid_thw,
                    second_per_grid_ts=[1.0]*len(grid_thw_rope),  # second_per_grid_ts,
                    attention_mask=torch.ones([1, len(current_text_ids)], dtype=torch.long, device=device),  # attention_mask, 全1掩码？
                )  # current_packed_position_ids: [3, 1, L]
                # mrope 上区分  ref image vae特征与video vae特征
                current_pos_ids = shift_position_ids(current_pos_ids, pos_shift = 1000, attn_modes = current_attn_modes, split_lens = current_split_lens, shift_attn_mode=['full_noise',"full"], pro_type = 10, i_sample_task=i_sample_task, i_sample_modality=i_sample_modality)

            if cfg_text_scale > 1.0:
                uncond_mask = i_sample_modality!=0
                _, uncond_pos_ids, _ = self.uncond_split_pro_kvcache(uncond_mask, current_text_ids, device, dtype, apply_qwen_2_5_vl_pos_emb, grid_thw_rope = grid_thw_rope, current_attn_modes=current_attn_modes, current_split_lens=current_split_lens, i_sample_task=i_sample_task, i_sample_modality=i_sample_modality)


            extra_inputs = {}  # {'mode': "und"}
            if self.use_moe: # NOTE: packed_und_token_indexes 的计算可能有问题，在x2v的任务中
                if N_vit_split != 0:
                    packed_und_token_indexes = torch.cat([current_text_indexes_local, current_vit_indexes_local], dim=0)
                else:
                    packed_und_token_indexes = current_text_indexes_local
                extra_inputs.update(
                    packed_und_token_indexes=packed_und_token_indexes.to(dtype=index_dtype),
                    packed_gen_token_indexes=current_vae_token_indexes_local.to(dtype=index_dtype),
                )

            timestep = torch.zeros(x_t.shape[0], device=x_t.device)
            timestep[current_vae_mse_indexes_local_in_vae] = torch.tensor([1.] * current_vae_mse_indexes_local_in_vae.shape[0], device=x_t.device)

            # --- 存入 视觉特征 编码 （vae condition）---
            timestep_embed = self.time_embedder(timestep)
            latent_pos_embed = self.latent_pos_embed(vae_position_ids)
            vae_embed = self.vae2llm(x_t) + timestep_embed + latent_pos_embed
            vae_embed = vae_embed.to(current_sequence.dtype)
            current_sequence[current_vae_token_indexes_local] = vae_embed

            ## ++++++++= for kv cache
            gen_context = self.init_gen_context(device=device, dtype=torch.int32) # gen_context: 初始化kv_lens，ropes，past_key_values
            cfg_text_context = deepcopy(gen_context)  #在添加当前文本之前使用，对应“没有包含这段新文本指令”的上下文，作为负向引导（negative prompt）使用，
            cfg_vision_context = deepcopy(gen_context )
            # current_cond_len = current_vae_mse_indexes_local[0]  # 处理到第一个 noise latent 前, 即前面的condition 共有 current_cond_len 个元素

            current_cond_start, current_cond_end = 0, 0

            self.language_model.eval()
            self.eval()
            for i_attn_mode_, current_cond_len in zip(current_attn_modes, current_split_lens):
                current_cond_end += current_cond_len
                if i_attn_mode_ == "noise": # 求解  noise latent 前 的 kv cache
                    vae_in_packed_sequence_index = torch.arange(current_cond_start, current_cond_end, dtype=torch.long, device=device) # vae split 在 packed_sequence 中的索引
                    packed_seqlens_vae = current_cond_len

                    # 生成用于 vae split 的 extra_inputs
                    target_packed_vae_token_indexes = torch.arange(1, current_cond_len-1, dtype=torch.long, device=device)
                    target_packed_text_indexes = torch.tensor([0, current_cond_len-1], dtype=torch.long, device=device)

                    break

                if i_attn_mode_ == 'causal':
                    is_causal = True
                else:
                    is_causal = False

                gen_context = self.update_gen_context(current_sequence, current_pos_ids, gen_context, extra_inputs, current_cond_start, current_cond_end, current_cond_len, device, dtype, is_causal = is_causal)
                if cfg_text_scale > 1.0 and i_sample_modality[current_cond_start] != 0:
                    cfg_text_context = self.update_gen_context(current_sequence, current_pos_ids, cfg_text_context, extra_inputs, current_cond_start, current_cond_end, current_cond_len, device, dtype, is_causal = is_causal)
                if cfg_vision_scale > 1.0 and i_sample_modality[current_cond_start] > 1: # i_sample_modality[current_cond_start] != 4 则为 cfg_vit
                    cfg_vision_context = self.update_gen_context(current_sequence, current_pos_ids, cfg_vision_context, extra_inputs, current_cond_start, current_cond_end, current_cond_len, device, dtype, is_causal = is_causal)

                current_cond_start = current_cond_end


            for _ in range(1):
                timestep = torch.zeros(x_t.shape[0], device=x_t.device)
                # for group-by-group generation

                for i, timestep_ in enumerate(timesteps):

                    timestep[current_vae_mse_indexes_local_in_vae] = torch.tensor([timestep_] * current_vae_mse_indexes_local_in_vae.shape[0], device=x_t.device)
                    if timestep_ > cfg_interval[0] and timestep_ <= cfg_interval[1]:
                        cfg_text_scale_ = cfg_text_scale
                        cfg_vision_scale_ = cfg_vision_scale # 默认 vit_uncond 和text_uncond 都采用 同一 cfg_interval
                    else:
                        cfg_text_scale_ = 1.0
                        cfg_vision_scale_ = 1.0

                    # --- 视觉特征 编码 ---
                    timestep_embed = self.time_embedder(timestep)
                    latent_pos_embed = self.latent_pos_embed(vae_position_ids)
                    vae_embed = self.vae2llm(x_t) + timestep_embed + latent_pos_embed
                    vae_embed = vae_embed.to(current_sequence.dtype)

                    # 一切正常再做赋值
                    current_sequence[current_vae_token_indexes_local] = vae_embed

                    # 提取出VAE部分的值
                    packed_sequence_vae = current_sequence[vae_in_packed_sequence_index]

                    extra_inputs_vae = {}
                    if self.use_moe:
                        extra_inputs_vae = {"mode": "gen", "packed_vae_token_indexes": target_packed_vae_token_indexes, "packed_text_indexes": target_packed_text_indexes}


                    v_t_output = self.language_model.forward_inference(
                        packed_query_sequence=packed_sequence_vae,  # [1026, 1536]
                        query_lens=torch.tensor([packed_seqlens_vae],dtype=torch.int32, device=device),  # [1]
                        packed_query_position_ids=current_pos_ids[:, :, current_cond_start:current_cond_end],  # [1026]
                        packed_query_indexes=vae_in_packed_sequence_index,  # [1026]
                        past_key_values=gen_context['past_key_values'],  # <class 'modeling.lance.qwen2_navit.NaiveCache'>
                        key_values_lens=gen_context['kv_lens'],  # [1]
                        packed_key_value_indexes=torch.arange(0,gen_context['kv_lens'][0], dtype=torch.int64, device=device),  # [76]
                        update_past_key_values=False,
                        is_causal=False,
                        **extra_inputs_vae,
                    )

                    v_t = self.llm2vae(v_t_output.packed_query_sequence)
                    v_t = v_t[target_packed_vae_token_indexes]

                    # --- 引入 cfg ---
                    if cfg_text_scale_ > 1.0:
                        cfg_text_output = self.language_model.forward_inference(
                            packed_query_sequence=packed_sequence_vae,
                            query_lens=torch.tensor([packed_seqlens_vae],dtype=torch.int32, device=device),
                            packed_query_position_ids=uncond_pos_ids[:,:,cfg_text_context['kv_lens'][0]:cfg_text_context['kv_lens'][0]+packed_seqlens_vae],
                            packed_query_indexes=vae_in_packed_sequence_index - sum(i_sample_modality==0), # 对应 packed_sequence_vae 在整个cfg序列中的index
                            past_key_values=cfg_text_context['past_key_values'],
                            key_values_lens=cfg_text_context['kv_lens'],
                            packed_key_value_indexes=torch.arange(0,cfg_text_context['kv_lens'][0], dtype=torch.int64, device=device),
                            update_past_key_values=False,
                            is_causal=False,
                            **extra_inputs_vae,
                        )
                        cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
                        cfg_text_v_t = cfg_text_v_t[target_packed_vae_token_indexes]

                        if cfg_vision_pro:
                            cfg_vision_output = self.language_model.forward_inference(
                                packed_query_sequence=packed_sequence_vae,
                                query_lens=torch.tensor([packed_seqlens_vae],dtype=torch.int32, device=device),
                                packed_query_position_ids=vision_uncond_pos_ids[:,:,cfg_vision_context['kv_lens'][0]:cfg_vision_context['kv_lens'][0]+packed_seqlens_vae],
                                packed_query_indexes=vae_in_packed_sequence_index - sum(i_sample_modality==4), # 对应 packed_sequence_vae 在整个cfg序列中的index
                                past_key_values=cfg_vision_context['past_key_values'],
                                key_values_lens=cfg_vision_context['kv_lens'],
                                packed_key_value_indexes=torch.arange(0,cfg_vision_context['kv_lens'][0], dtype=torch.int64, device=device),
                                update_past_key_values=False,
                                is_causal=False,
                                **extra_inputs_vae,
                            )

                            cfg_text_vision_v_t = self.llm2vae(cfg_vision_output.packed_query_sequence)
                            cfg_text_vision_v_t = cfg_text_vision_v_t[target_packed_vae_token_indexes]

                            v_t_ = cfg_text_vision_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t) + cfg_vision_scale_  * (cfg_text_v_t - cfg_text_vision_v_t)
                        else:
                            v_t_ = cfg_text_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t)

                        # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                        if cfg_renorm_type == "global":
                            norm_v_t = torch.norm(v_t)
                            norm_v_t_ = torch.norm(v_t_)
                            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        elif cfg_renorm_type == "channel":
                            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                            norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        elif cfg_renorm_type.lower() in ("", "none", "null"):
                            scale = 1
                        else:
                            raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                        v_t = v_t_ * scale

                    x_t[current_vae_mse_indexes_local_in_vae] = x_t[current_vae_mse_indexes_local_in_vae] - v_t.to(x_t.device) * dts[i]  # velocity pointing from data to noise

            # ---- 每个样本各自重排到 [T,H,W,C]，避免用最后一个样本的 t/h/w 去重排整批 ----
            curr_seq_target, patch = 0, []
            for i_target in range(N_noise_element):

                pt, ph, pw = self.latent_patch_size
                t, h, w = vid_shape_list[i_target]
                len_target = t * h * w

                x_t_ =  rearrange(x_t[curr_seq_target : curr_seq_target + len_target], "(t h w) (pt ph pw c) -> (t pt) (h ph) (w pw) c", t=t, h=h, w=w, pt=pt, ph=ph, pw=pw)

                patch.append(x_t_)
                curr_seq_target += len_target
            # patch = torch.cat(patch, dim=0)
            x_t_all.append(patch)
            # ✅ 成功处理一个 GEN 样本，推进 gen_idx
            gen_idx += 1

        # self.logger.info(f"Validation step done with {len(x_t_all)} samples.")

        if caption != None:
            return x_t_all, [caption], padded_videos, index

        # if padded_videos != []:
        #     return x_t_all, padded_videos

        return x_t_all

    def get_uncond_attn_modes_split_lens(self, current_attn_modes, current_split_lens, uncond_mask):
        # 根据 uncond_mask 过滤出无条件样本的部分
        curr = 0
        uncond_attn_modes, uncond_split_lens = [], []
        for i, split_len in enumerate(current_split_lens):

            # 检查当前 split 对应的 mask 是否全为 True
            mask_slice = uncond_mask[curr:curr+split_len]
            if mask_slice.all():
                uncond_attn_modes.append(current_attn_modes[i])
                uncond_split_lens.append(split_len)

            # 更新当前位置
            curr += split_len

        return uncond_attn_modes, uncond_split_lens




    def uncond_split_pro_kvcache(
        self,
        uncond_mask,
        current_text_ids,
        device,
        dtype,
        apply_qwen_2_5_vl_pos_emb=False,
        uncond_pos_ids=None,
        grid_thw_rope=None,
        current_attn_modes=None,
        current_split_lens=None,
        i_sample_task=None,
        i_sample_modality=None,
    ):
        """
        uncond_attn_modes: uncond 序列的 attn_mode 列表
        uncond_split_lens: uncond 序列的 split_len 列表
        uncond_packed_gen_token_indexes: uncond 序列中 gen_token 索引列表 (for extra_inputs)
        uncond_packed_und_token_indexes: uncond 序列中 und_token 索引列表 (for extra_inputs)
        uncond_seq_len: uncond 序列的长度
        uncond_pad: uncond 序列的 pad 长度
        """
        # 提取 uncond 序列的 text_ids
        uncond_text_ids = current_text_ids[uncond_mask]
        uncond_seq_len = len(uncond_text_ids)


        if apply_qwen_2_5_vl_pos_emb:
            uncond_pos_ids, uncond_rope_deltas = self.language_model.get_rope_index(
                input_ids=uncond_text_ids.unsqueeze(0),
                image_grid_thw=grid_thw_rope,  # vae_video_grid_thw[gen_idx : gen_idx + 1],
                video_grid_thw=grid_thw_rope,  # vae_video_grid_thw[gen_idx : gen_idx + 1],  # video_grid_thw,
                second_per_grid_ts=[1.0] * len(grid_thw_rope),  # second_per_grid_ts,
                attention_mask=torch.ones([1, len(uncond_text_ids)], dtype=torch.long, device=device),  # attention_mask, 全1掩码？
            )
            # mrope 上区分  ref image vae特征与video vae特征
            uncond_attn_modes, uncond_split_lens = self.get_uncond_attn_modes_split_lens( current_attn_modes, current_split_lens, uncond_mask)
            i_sample_task = i_sample_task[uncond_mask]
            i_sample_modality = i_sample_modality[uncond_mask]

            uncond_pos_ids = shift_position_ids(uncond_pos_ids, pos_shift = 1000, attn_modes = uncond_attn_modes, split_lens = uncond_split_lens, shift_attn_mode=['full_noise',"full"], pro_type = 10, i_sample_task=i_sample_task, i_sample_modality=i_sample_modality)
        else:
            uncond_pos_ids = torch.tensor(uncond_pos_ids, dtype=torch.long, device=device)[:uncond_seq_len]

        return (
            uncond_text_ids,
            uncond_pos_ids,
            uncond_seq_len,
        )



    def update_gen_context(self, current_sequence, current_pos_ids, gen_context, extra_inputs, current_cond_start, current_cond_end, current_cond_len, device, dtype, is_causal = True):
        extra_inputs_cond = {}
        extra_inputs_gen_mask = (extra_inputs["packed_gen_token_indexes"] >= current_cond_start) & (extra_inputs["packed_gen_token_indexes"] < current_cond_end)
        extra_inputs_cond["packed_vae_token_indexes"] = extra_inputs["packed_gen_token_indexes"][extra_inputs_gen_mask] - gen_context['kv_lens']
        extra_inputs_und_mask = (extra_inputs["packed_und_token_indexes"] >= current_cond_start) & (extra_inputs["packed_und_token_indexes"] < current_cond_end)
        extra_inputs_cond["packed_text_indexes"] = extra_inputs["packed_und_token_indexes"][extra_inputs_und_mask] - gen_context['kv_lens']

        if extra_inputs_cond["packed_vae_token_indexes"].shape[0] > 0 :
            mode_ = "gen"
        else:
            mode_ = "und"

        # 只对condition 求解kv cache
        output = self.language_model.forward_inference(
            packed_query_sequence=current_sequence[current_cond_start:current_cond_end],
            query_lens=torch.tensor([current_cond_len],dtype=torch.int32, device=device),
            packed_query_position_ids=current_pos_ids[:, :, current_cond_start:current_cond_end],
            #packed_query_indexes=torch.arange(current_cond_start,current_cond_end, dtype=torch.long, device=device), # 放置 当前新输入 的位置。
            packed_query_indexes=torch.arange(gen_context['kv_lens'][0],gen_context['kv_lens'][0] + current_cond_len, dtype=torch.long, device=device), # 放置 当前新输入 的位置。
            past_key_values=gen_context['past_key_values'],
            packed_key_value_indexes=torch.arange(0,gen_context['kv_lens'][0], dtype=torch.int64, device=device), # 放置 历史缓存 (Past KV) 的位置。
            key_values_lens=gen_context['kv_lens'], # torch.tensor([0], dtype=torch.int32),
            update_past_key_values=True,
            is_causal=is_causal,
            mode = mode_,
            **extra_inputs_cond
        )

        gen_context['past_key_values'] = output.past_key_values
        gen_context['kv_lens'] += current_cond_len #= torch.tensor([int(current_cond_end)], dtype=torch.int32, device=device)
        #gen_context['ropes'] = [int(current_cond_end)]

        return gen_context
