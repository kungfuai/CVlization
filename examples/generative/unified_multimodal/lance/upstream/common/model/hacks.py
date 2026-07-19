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

def hack_qwen2_5_vl_config(language_model):
    # HACK!!!!!
    language_model.config.image_token_id = 151655
    language_model.config.video_token_id = 151656
    language_model.config.vision_start_token_id = 151652
    language_model.config.vision_end_token_id = 151653

    language_model.config.vision_config = {
        "depth": 32,
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_chans": 3,
        "out_hidden_size": 2048,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "window_size": 112,
        "fullatt_block_indexes": [
            7,
            15,
            23,
            31
        ],
        "tokens_per_second": 2,
        "temporal_patch_size": 2
    }

    language_model.config.rope_scaling = {
        "type": "mrope",
        "mrope_section": [
            16,
            24,
            24
        ]
    }

    return language_model
