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

"""
配置定义与轻量工厂函数。

- TemplateArguments  : 对话模板
- ModelArguments     : 模型结构/加载参数
- DataArguments      : 验证集输入参数
- TrainingArguments  : 推理运行期仍会使用的模型加载/采样/兼容字段
- InferenceArguments : 推理专用参数（继承 TrainingArguments）
- EvaluationArguments: 评估专用参数（继承 InferenceArguments）

当前模块同时负责：
- 定义训练/推理共用的 dataclass
- 解析 `path_default.yaml`
- 为推理链提供轻量配置工厂能力
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import yaml

# ==============================================
# 模型路径配置管理
# ==============================================

# 全局缓存，避免重复加载
_MODEL_PATH_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_DEFAULT_PATH_FILE = Path(__file__).with_name("path_default.yaml")
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _get_nested_value(config: Dict[str, Any], path_key: str) -> Any:
    """
    根据点分路径从嵌套配置中取值，例如 "vit.qwen2_5_vl"。
    """
    value: Any = config
    for key in path_key.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ValueError(f"Path key '{path_key}' not found in {_DEFAULT_PATH_FILE.name}")
    return value


def _resolve_config_values(value: Any, config: Dict[str, Any]) -> Any:
    """
    递归解析配置中的占位符，保持原有的嵌套结构不变。
    """
    if isinstance(value, dict):
        return {k: _resolve_config_values(v, config) for k, v in value.items()}
    if isinstance(value, str):
        return _resolve_placeholders(value, config)
    return value


def _resolve_placeholders(path: str, config: Dict[str, Any]) -> str:
    """
    递归解析路径中的占位符，例如 ${base_dir} 或 ${vit.qwen2_5_vl}
    """
    matches = _PLACEHOLDER_PATTERN.findall(path)
    
    if not matches:
        return path
    
    result = path
    for match in matches:
        try:
            value = _get_nested_value(config, match)
        except ValueError as exc:
            raise ValueError(f"Placeholder ${match} not found in {_DEFAULT_PATH_FILE.name}") from exc

        # 递归解析值中的占位符
        resolved_value = _resolve_placeholders(str(value), config)
        result = result.replace(f"${{{match}}}", resolved_value)

    return result


def get_model_path_config(reload: bool = False) -> Dict[str, Any]:
    """
    加载并解析 path_default.yaml 配置文件
    :param reload: 强制重新加载，忽略缓存
    :return: 解析后的配置字典
    """
    global _MODEL_PATH_CONFIG_CACHE
    
    if _MODEL_PATH_CONFIG_CACHE is not None and not reload:
        return _MODEL_PATH_CONFIG_CACHE
    
    if not _DEFAULT_PATH_FILE.exists():
        raise FileNotFoundError(
            f"Model path configuration file not found: {_DEFAULT_PATH_FILE}"
        )

    with _DEFAULT_PATH_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved_config = _resolve_config_values(config, config)
    _MODEL_PATH_CONFIG_CACHE = resolved_config

    return resolved_config


def get_model_path(path_key: str) -> str:
    """
    获取指定的路径值
    :param path_key: 路径键，支持嵌套，例如 "vit.qwen2_5_vl", "data.t2i"
    :return: 解析后的完整路径
    """
    config = get_model_path_config()
    value = _get_nested_value(config, path_key)

    return str(value) if value is not None else ""

@dataclass
class TemplateArguments:
    chat_template: List[str] = (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'Describe this image.<|im_end|>\n<|im_start|>assistant\n',
    )  # NOTE: instruction 需要考虑适配不同数据类型；模板中间插入 VIT token，最后插入 text token
    chat_template_T2I: List[str] = (
        '<|im_start|>system\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n<|quad_start|><|im_end|>\n<|im_start|>assistant\n',
    )  # NOTE: 模板中间插入 text token，最后插入 VAE token
    pad_token_template_T2I: str = "<|quad_start|>"
    pad_token_template: str = "<|quad_end|>"


@dataclass
class ModelArguments:
    model_path:                 str = ""
    llm_path:                   str = ""
    llm_qk_norm:                bool = True
    llm_qk_norm_und:            bool = True
    llm_qk_norm_gen:            bool = True
    tie_word_embeddings:        bool = False
    layer_module:               str = "Qwen2MoTDecoderLayer"
    vit_path:                   str = ""
    max_num_frames:             int = 25
    max_latent_size:            int = 64
    latent_patch_size:          List[int] = (1, 2, 2)  # pt ph pw
    vit_patch_size:             int = 14
    vit_patch_size_temporal:    int = 2
    vit_max_num_patch_per_side: int = 70
    connector_act:              str = "gelu_pytorch_tanh"
    interpolate_pos:            bool = False
    vit_select_layer:           int = -2
    vit_rope:                   bool = False

    text_cond_dropout_prob:     float = 0.1
    vae_cond_dropout_prob:      float = 0.3
    vit_cond_dropout_prob:      float = 0.3
    vit_type:                   str = "qwen2_5_vl"  # options: qwen2_5_vl

    val_text_cond_dropout_prob: float = 0
    val_vae_cond_dropout_prob:  float = 0
    val_vit_cond_dropout_prob:  float = 0

    cfg_text_scale:             float = 4.0  # for validation


@dataclass
class DataArguments:
    val_dataset_config_file:    Optional[str] = None


@dataclass
class TrainingArguments:
    # 推理运行期开关
    apply_chat_template:        bool = False  # 是否对输入文本套用 Qwen2.5-VL chat template
    apply_qwen_2_5_vl_pos_emb:  bool = False  # 是否启用 Qwen2.5-VL position embedding

    vae_model_type:             str = "seedance"
    visual_gen:                 bool = True
    visual_und:                 bool = True
    freeze_und:                 bool = False
    copy_init_moe:              bool = False
    finetune_from_hf:           bool = False
    finetune_from_vlm:          bool = False
    use_flex:                   bool = False
    num_replicate:              int = 1
    num_shard:                  int = 1

    global_seed:                int = 2025

    # 采样相关
    timestep_shift:             float = 1.0
    validation_data_seed:       int = 42
    validation_num_timesteps:   int = 30
    validation_timestep_shift:  float = 3.0
    validation_max_samples:     int = 8
    validation_noise_seed:      int = 2025
    validation_video_saving_fps:int = 12
    validation_log_type:        str = "direct"

    # CFG 与文本条件控制
    cfg_type:                   int = 0       # 0: 完全去除文本条件; 1: 仅保留特殊 token; 2: 保留特殊 token + 中间文本 token 替换为 <NULL>
    cfg_uncond_token_id:        int = 151643  # 仅在 cfg_type=2 时生效
    cfg_interval:               List[float] = field(default_factory=lambda: [0.4, 1.0])
    cfg_renorm_min:             float = 0
    cfg_renorm_type:            str = "global"  # global | channel | ""

    # 额外 embedding 开关
    use_task_embedding:         bool = False
    use_modality_embedding:     bool = False


@dataclass
class InferenceArguments(TrainingArguments):
    save_path_gen:              str = "tmp/results/inference/generation"  # 生成视频/图像保存路径
    save_path_gt:               str = ""    # ground truth 视频/图像保存路径，默认不保存
    video_height:               int = 480
    video_width:                int = 480
    num_frames:                 int = 50
    task:                       str = "t2v"  # t2v / t2i / edit / idip ...
    resolution:                 str = "video_360p"  # image_256res, image_512res, video_192p, video_360p 等
    text_template:              bool = False  # 是否使用 system_prompt 文本模板
    max_duration:               float = 6.0  # 最大视频时长（秒）

    system_prompt_type:         str = "SP0"  # options: SP1, SP2 ...
    use_KVcache:                bool = False


@dataclass
class EvaluationArguments(InferenceArguments):
    config_json_path:           str = field(default="", metadata={"help": "配置 JSON 文件路径"})  # 提供 config 则用其覆盖参数
    sample_num_per_prompt:      int = field(default=4, metadata={"help": "每个 case 的采样数量"})
    max_eval_cases:             int = field(default=0, metadata={"help": "限制评测 case 数量，0 表示全量"})
    do_sample:                  bool = False  # UND 任务是否使用采样策略
    evaluation_seed:            int = 42
    quick_debug:                bool = False  # 快速调试模式
