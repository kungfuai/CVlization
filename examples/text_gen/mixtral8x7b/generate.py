from pathlib import Path
from glob import glob
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging

import sys
sys.path.append("mixtral-offloading")
from src.build_model import OffloadConfig, QuantConfig, build_model

from subprocess import run
run("huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo".split())

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

assert Path("/root", ".cache", "huggingface", "hub", "models--lavawolfiee--Mixtral-8x7B-Instruct-v0.1-offloading-demo").exists()
pretrain_path = "/root/.cache/huggingface/hub/models--" + quantized_model_name.replace("/", "--")
snapshots = glob(str(Path(pretrain_path, "snapshots", "*", "config.json")))
local_model_dir = Path(snapshots[0]).parent
config = AutoConfig.from_pretrained(local_model_dir)
state_path = local_model_dir

device = torch.device("cuda:0")

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
# offload_per_layer = 0  # consumes 20.4 GB
# offload_per_layer = 1  # consumes 18 GB
# offload_per_layer = 2  # consumes 16.9 GB
# offload_per_layer = 3 # consumes 14.1 GB
# offload_per_layer = 4  # consumes 12.2 GB
offload_per_layer = 5  # consumes 10.2 GB
###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)


from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None

seq_len = 0

if True:
  user_input = "tell me a joke"

  user_entry = dict(role="user", content=user_input)
  input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

  if past_key_values is None:
    attention_mask = torch.ones_like(input_ids)
  else:
    seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
    attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

  print("Mixtral: ", end="")
  result = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    streamer=streamer,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_hidden_states=True,
  )
  print("\n")

  sequence = result["sequences"]
  past_key_values = result["past_key_values"]