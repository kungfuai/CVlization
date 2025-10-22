import os
import time
import argparse
from mmgp import offload, safetensors2, profile_type 
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import gradio as gr
import random
import json
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video
from wan.modules.attention import get_attention_modes
import torch
import gc
import traceback
import math
import asyncio

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")

    parser.add_argument(
        "--quantize-transformer",
        action="store_true",
        help="On the fly 'transformer' quantization"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a shared URL to access webserver remotely"
    )

    parser.add_argument(
        "--lock-config",
        action="store_true",
        help="Prevent modifying the configuration from the web interface"
    )

    parser.add_argument(
        "--preload",
        type=str,
        default="0",
        help="Megabytes of the diffusion model to preload in VRAM"
    )

    parser.add_argument(
        "--multiple-images",
        action="store_true",
        help="Allow inputting multiple images with image to video"
    )


    parser.add_argument(
        "--lora-dir-i2v",
        type=str,
        default="loras_i2v",
        help="Path to a directory that contains Loras for i2v"
    )


    parser.add_argument(
        "--lora-dir",
        type=str,
        default="loras", 
        help="Path to a directory that contains Loras"
    )


    parser.add_argument(
        "--lora-preset",
        type=str,
        default="",
        help="Lora preset to preload"
    )

    parser.add_argument(
        "--lora-preset-i2v",
        type=str,
        default="",
        help="Lora preset to preload for i2v"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=-1,
        help="Profile No"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default=1,
        help="Verbose level"
    )

    parser.add_argument(
        "--server-port",
        type=str,
        default=0,
        help="Server port"
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default="",
        help="Server name"
    )

    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open browser"
    )

    parser.add_argument(
        "--t2v",
        action="store_true",
        help="text to video mode"
    )

    parser.add_argument(
        "--i2v",
        action="store_true",
        help="image to video mode"
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable pytorch compilation"
    )

    # parser.add_argument(
    #     "--fast",
    #     action="store_true",
    #     help="use Fast model"
    # )

    # parser.add_argument(
    #     "--fastest",
    #     action="store_true",
    #     help="activate the best config"
    # )

    parser.add_argument(
    "--attention",
    type=str,
    default="",
    help="attention mode"
    )

    parser.add_argument(
    "--vae-config",
    type=str,
    default="",
    help="vae config mode"
    )    


    args = parser.parse_args()

    return args

attention_modes_supported = get_attention_modes()

args = _parse_args()
args.flow_reverse = True


lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

preload =int(args.preload)
force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
quantizeTransformer = args.quantize_transformer

transformer_choices_t2v=["ckpts/wan2.1_text2video_1.3B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_bf16.safetensors", "ckpts/wan2.1_text2video_14B_quanto_int8.safetensors"]   
transformer_choices_i2v=["ckpts/wan2.1_image2video_480p_14B_bf16.safetensors", "ckpts/wan2.1_image2video_480p_14B_quanto_int8.safetensors", "ckpts/wan2.1_image2video_720p_14B_bf16.safetensors", "ckpts/wan2.1_image2video_720p_14B_quanto_int8.safetensors"]
text_encoder_choices = ["ckpts/models_t5_umt5-xxl-enc-bf16.safetensors", "ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors"]

server_config_filename = "gradio_config.json"

if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "auto",  
                     "transformer_filename": transformer_choices_t2v[0], 
                     "transformer_filename_i2v": transformer_choices_i2v[1],  ########
                     "text_encoder_filename" : text_encoder_choices[1],
                     "compile" : "",
                     "default_ui": "t2v",
                     "vae_config": 0,
                     "profile" : profile_type.LowRAM_LowVRAM }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)


transformer_filename_t2v = server_config["transformer_filename"]
transformer_filename_i2v = server_config.get("transformer_filename_i2v", transformer_choices_i2v[1]) ########

text_encoder_filename = server_config["text_encoder_filename"]
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

default_ui = server_config.get("default_ui", "t2v") 
use_image2video = default_ui != "t2v"
if args.t2v:
    use_image2video = False
if args.i2v:
    use_image2video = True

if use_image2video:
    lora_dir =args.lora_dir_i2v
    lora_preselected_preset = args.lora_preset_i2v
else:
    lora_dir =args.lora_dir
    lora_preselected_preset = args.lora_preset

default_tea_cache = 0
# if args.fast : #or args.fastest
#     transformer_filename_t2v = transformer_choices_t2v[2]
#     attention_mode="sage2" if "sage2" in attention_modes_supported else "sage"
#     default_tea_cache = 0.15
#     lock_ui_attention = True
#     lock_ui_transformer = True

if  args.compile: #args.fastest or
    compile="transformer"
    lock_ui_compile = True


#attention_mode="sage"
#attention_mode="sage2"
#attention_mode="flash"
#attention_mode="sdpa"
#attention_mode="xformers"
# compile = "transformer"

def download_models(transformer_filename, text_encoder_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        
    
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/Wan2.1" 
    sourceFolderList = ["xlm-roberta-large", "",  ]
    fileList = [ [], ["Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ] + computeList(text_encoder_filename) + computeList(transformer_filename) ]   
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
             for onefile in files:     
                if len(sourceFolder) > 0: 
                    if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                else:
                    if not os.path.isfile(targetRoot + onefile ):          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)


offload.default_verboseLevel = verbose_level

download_models(transformer_filename_i2v if use_image2video else transformer_filename_t2v, text_encoder_filename) 

def sanitize_file_name(file_name):
    return file_name.replace("/","").replace("\\","").replace(":","").replace("|","").replace("?","").replace("<","").replace(">","").replace("\"","") 

def extract_preset(lset_name, loras):
    lset_name = sanitize_file_name(lset_name)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 

    if not os.path.isfile(lset_name_filename):
        raise gr.Error(f"Preset '{lset_name}' not found ")

    with open(lset_name_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    lset = json.loads(text)

    loras_choices_files = lset["loras"]
    loras_choices = []
    missing_loras = []
    for lora_file in loras_choices_files:
        loras_choice_no = loras.index(os.path.join(lora_dir, lora_file))
        if loras_choice_no < 0:
            missing_loras.append(lora_file)
        else:
            loras_choices.append(str(loras_choice_no))

    if len(missing_loras) > 0:
        raise gr.Error(f"Unable to apply Lora preset '{lset_name} because the following Loras files are missing: {missing_loras}")
    
    loras_mult_choices = lset["loras_mult"]
    return loras_choices, loras_mult_choices

def setup_loras(pipe,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    loras_names = []
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []

    from pathlib import Path

    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")

    default_lora_preset = ""

    if lora_dir != None:
        import glob
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets =  glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets]

    if len(loras) > 0:
        loras_names = [ Path(lora).stem for lora in loras  ]
        offload.load_loras_into_model(pipe["transformer"], loras,  activate_all_loras=False, split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str= extract_preset(default_lora_preset, loras)

    return loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets


def load_t2v_model(model_filename, value):

    cfg = WAN_CONFIGS['t2v-14B']
    # cfg = WAN_CONFIGS['t2v-1.3B']    
    print("load t2v model...")

    wan_model = wan.WanT2V(
        config=cfg,
        checkpoint_dir="ckpts",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        model_filename=model_filename,
        text_encoder_filename= text_encoder_filename
    )

    pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "vae": wan_model.vae.model } 

    return wan_model, pipe

def load_i2v_model(model_filename, value):


    if value == '720P':
        print("load 14B-720P i2v model...")
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="ckpts",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p= True,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )            
        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "text_encoder_2": wan_model.clip.model, "vae": wan_model.vae.model } #

    if value == '480P':
        print("load 14B-480P i2v model...")
        cfg = WAN_CONFIGS['i2v-14B']
        wan_model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="ckpts",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p= False,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename

        )
        pipe = {"transformer": wan_model.model, "text_encoder" : wan_model.text_encoder.model,  "text_encoder_2": wan_model.clip.model, "vae": wan_model.vae.model } #

    return wan_model, pipe

def load_models(i2v,  lora_dir,  lora_preselected_preset ):
    download_models(transformer_filename_i2v if i2v else transformer_filename_t2v, text_encoder_filename) 

    if i2v:
        res720P= "720p" in transformer_filename_i2v
        wan_model, pipe = load_i2v_model(transformer_filename_i2v,"720P" if res720P else "480P")
    else:
        wan_model, pipe = load_t2v_model(transformer_filename_t2v,"")

    kwargs = { "extraModelsToQuantize": None}
    if profile == 2 or profile == 4:
        kwargs["budgets"] = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100, "*" : 1000 }
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }


    loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets = setup_loras(pipe,  lora_dir, lora_preselected_preset, None)
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = quantizeTransformer, **kwargs)  


    return wan_model, offloadobj, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets

wan_model, offloadobj,  loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets = load_models(use_image2video, lora_dir, lora_preselected_preset )
gen_in_progress = False

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def get_default_flow(model_filename):
    return 3.0 if "480p" in model_filename else 5.0 

def generate_header(model_filename, compile, attention_mode):
    header = "<H2 ALIGN=CENTER><SPAN> ----------------- "
    
    if "image" in model_filename:
        model_name = "Wan2.1 image2video"
        model_name += "720p" if "720p" in model_filename else "480p"
    else:
        model_name = "Wan2.1 text2video"
        model_name += "14B" if "14B" in model_filename else "1.3B"

    header += model_name 
    header += " (attention mode: " + (attention_mode if attention_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_supported:
        header += " -NOT INSTALLED-"

    if compile:
        header += ", pytorch compilation ON"
    header += ") -----------------</SPAN></H2>"

    return header

def apply_changes(  state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
                    vae_config_choice,
                    default_ui_choice ="t2v",
):
    if args.lock_config:
        return
    if gen_in_progress:
        yield "<DIV ALIGN=CENTER>Unable to change config when a generation is in progress</DIV>"
        return
    global offloadobj, wan_model, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_filename": transformer_choices_t2v[transformer_t2v_choice], 
                     "transformer_filename_i2v": transformer_choices_i2v[transformer_i2v_choice],  ##########
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "compile" : compile_choice,
                     "profile" : profile_choice,
                     "vae_config" : vae_config_choice,
                     "default_ui" : default_ui_choice,
                       }

    if Path(server_config_filename).is_file():
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        old_server_config = json.loads(text)
        if lock_ui_transformer:
            server_config["transformer_filename"] = old_server_config["transformer_filename"]
            server_config["transformer_filename_i2v"] = old_server_config["transformer_filename_i2v"]
        if lock_ui_attention:
            server_config["attention_mode"] = old_server_config["attention_mode"]
        if lock_ui_compile:
            server_config["compile"] = old_server_config["compile"]

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    changes = []
    for k, v in server_config.items():
        v_old = old_server_config.get(k, None)
        if v != v_old:
            changes.append(k)

    state["config_changes"] = changes
    state["config_new"] = server_config
    state["config_old"] = old_server_config

    global attention_mode, profile, compile, transformer_filename_t2v, transformer_filename_i2v, text_encoder_filename, vae_config
    attention_mode = server_config["attention_mode"]
    profile = server_config["profile"]
    compile = server_config["compile"]
    transformer_filename_t2v = server_config["transformer_filename"]
    transformer_filename_i2v = server_config["transformer_filename_i2v"]
    text_encoder_filename = server_config["text_encoder_filename"]
    vae_config = server_config["vae_config"]

    if  all(change in ["attention_mode", "vae_config", "default_ui"] for change in changes ):
        if "attention_mode" in changes:
            pass

    else:
        wan_model = None
        offloadobj.release()
        offloadobj = None
        yield "<DIV ALIGN=CENTER>Please wait while the new configuration is being applied</DIV>"

        wan_model, offloadobj,  loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets = load_models(use_image2video, lora_dir,  lora_preselected_preset )


    yield "<DIV ALIGN=CENTER>The new configuration has been succesfully applied</DIV>"

    # return "<DIV ALIGN=CENTER>New Config file created. Please restart the Gradio Server</DIV>"

def update_defaults(state, num_inference_steps,flow_shift):
    if "config_changes" not in state:
        return get_default_flow("")
    changes = state["config_changes"] 
    server_config = state["config_new"] 
    old_server_config = state["config_old"] 

    if not use_image2video:
        old_is_14B = "14B" in server_config["transformer_filename"]
        new_is_14B = "14B" in old_server_config["transformer_filename"]

        trans_file = server_config["transformer_filename"]
        # if old_is_14B != new_is_14B:
        #     num_inference_steps, flow_shift = get_default_flow(trans_file)
    else:
        old_is_720P = "720P" in server_config["transformer_filename_i2v"]
        new_is_720P = "720P" in old_server_config["transformer_filename_i2v"]
        trans_file = server_config["transformer_filename_i2v"]
        if old_is_720P != new_is_720P:
            num_inference_steps, flow_shift = get_default_flow(trans_file)

    header = generate_header(trans_file, server_config["compile"], server_config["attention_mode"] )
    return num_inference_steps, flow_shift, header 


from moviepy.editor import ImageSequenceClip
import numpy as np

def save_video(final_frames, output_path, fps=24):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path, verbose= False, logger = None)

def build_callback(state, pipe, progress, status, num_inference_steps):
    def callback(step_idx, latents):
        step_idx += 1         
        if state.get("abort", False):
            # pipe._interrupt = True
            status_msg = status + " - Aborting"    
        elif step_idx  == num_inference_steps:
            status_msg = status + " - VAE Decoding"    
        else:
            status_msg = status + " - Denoising"   

        progress( (step_idx , num_inference_steps) , status_msg  ,  num_inference_steps)
            
    return callback

def abort_generation(state):
    if "in_progress" in state:
        state["abort"] = True
        wan_model._interrupt= True
        return gr.Button(interactive=  False)
    else:
        return gr.Button(interactive=  True)

def refresh_gallery(state):
    file_list = state.get("file_list", None)      
    return file_list
        
def finalize_gallery(state):
    choice = 0
    if "in_progress" in state:
        del state["in_progress"]
        choice = state.get("selected",0)
    
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True)

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    if data!=None:
        state["selected"] = data.get("index",0)
    return 

def expand_slist(slist, num_inference_steps ):
    new_slist= []
    inc =  len(slist) / num_inference_steps 
    pos = 0
    for i in range(num_inference_steps):
        new_slist.append(slist[ int(pos)])
        pos += inc
    return new_slist


def generate_video(
    prompt,
    negative_prompt,    
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    tea_cache,
    tea_cache_start_step_perc,    
    loras_choices,
    loras_mult_choices,
    image_to_continue,
    video_to_continue,
    max_frames,
    RIFLEx_setting,
    state,
    progress=gr.Progress() #track_tqdm= True

):
    
    from PIL import Image
    import numpy as np
    import tempfile


    if wan_model == None:
        raise gr.Error("Unable to generate a Video while a new configuration is being applied.")
    if attention_mode == "auto":
        attn = get_auto_attention()
    elif attention_mode in attention_modes_supported:
        attn = attention_mode
    else:
        raise gr.Error(f"You have selected attention mode '{attention_mode}'. However it is not installed on your system. You should either install it or switch to the default 'sdpa' attention.")

    width, height = resolution.split("x")
    width, height = int(width), int(height)


    if use_image2video:
        if "480p" in  transformer_filename_i2v and width * height > 848*480:
            raise gr.Error("You must use the 720P image to video model to generate videos with a resolution equivalent to 720P")

        resolution = str(width) + "*" + str(height)  
        if  resolution not in ['720*1280', '1280*720', '480*832', '832*480']:
            raise gr.Error(f"Resolution {resolution} not supported by image 2 video")


    else:
        if "1.3B" in  transformer_filename_t2v and width * height > 848*480:
            raise gr.Error("You must use the 14B text to video model to generate videos with a resolution equivalent to 720P")

    
    offload.shared_state["_attention"] =  attn
 
     # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if vae_config == 0:
        if device_mem_capacity >= 24000:
            use_vae_config = 1            
        elif device_mem_capacity >= 8000:
            use_vae_config = 2
        else:          
            use_vae_config = 3
    else:
        use_vae_config = vae_config

    if use_vae_config == 1:
        VAE_tile_size = 0  
    elif use_vae_config == 2:
        VAE_tile_size = 256  
    else: 
        VAE_tile_size = 128  


    global gen_in_progress
    gen_in_progress = True
    temp_filename = None
    if len(prompt) ==0:
        return
    prompts = prompt.replace("\r", "").split("\n")

    if use_image2video:
        if image_to_continue is not None:
            if isinstance(image_to_continue, list):
                image_to_continue = [ tup[0] for tup in image_to_continue ]
            else:
                image_to_continue = [image_to_continue]
            if len(prompts) >= len(image_to_continue):
                if len(prompts) % len(image_to_continue) !=0:
                    raise gr.Error("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                rep = len(prompts) // len(image_to_continue)
                new_image_to_continue = []
                for i, _ in enumerate(prompts):
                    new_image_to_continue.append(image_to_continue[i//rep] )
                image_to_continue = new_image_to_continue 
            else: 
                if len(image_to_continue) % len(prompts)  !=0:
                    raise gr.Error("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                rep = len(image_to_continue) // len(prompts)  
                new_prompts = []
                for i, _ in enumerate(image_to_continue):
                    new_prompts.append(  prompts[ i//rep] )
                prompts = new_prompts

        elif video_to_continue != None and len(video_to_continue) >0 :
            input_image_or_video_path = video_to_continue
            # pipeline.num_input_frames = max_frames
            # pipeline.max_frames = max_frames
        else:
            return
    else:
        input_image_or_video_path = None


    if len(loras) > 0:
        def is_float(element: any) -> bool:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
        list_mult_choices_nums = []
        if len(loras_mult_choices) > 0:
            list_mult_choices_str = loras_mult_choices.split(" ")
            for i, mult in enumerate(list_mult_choices_str):
                mult = mult.strip()
                if "," in mult:
                    multlist = mult.split(",")
                    slist = []
                    for smult in multlist:
                        if not is_float(smult):                
                            raise gr.Error(f"Lora sub value no {i+1} ({smult}) in Multiplier definition '{multlist}' is invalid")
                        slist.append(float(smult))
                    slist = expand_slist(slist, num_inference_steps )
                    list_mult_choices_nums.append(slist)
                else:
                    if not is_float(mult):                
                        raise gr.Error(f"Lora Multiplier no {i+1} ({mult}) is invalid")
                    list_mult_choices_nums.append(float(mult))
        if len(list_mult_choices_nums ) < len(loras_choices):
            list_mult_choices_nums  += [1.0] * ( len(loras_choices) - len(list_mult_choices_nums ) )

        offload.activate_loras(wan_model.model, loras_choices, list_mult_choices_nums)

    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
    state["selected"] = 0
 
    enable_RIFLEx = RIFLEx_setting == 0 and video_length > (6* 16) or RIFLEx_setting == 1
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576


   # TeaCache   
    trans = wan_model.model
    trans.enable_teacache = tea_cache > 0
 
    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    file_list = []
    state["file_list"] = file_list    
    from einops import rearrange
    save_path = os.path.join(os.getcwd(), "gradio_outputs")
    os.makedirs(save_path, exist_ok=True)
    video_no = 0
    total_video =  repeat_generation * len(prompts)
    abort = False
    start_time = time.time()
    for prompt in prompts:
        for _ in range(repeat_generation):
            if abort:
                break

            if trans.enable_teacache:
                trans.teacache_counter = 0
                trans.rel_l1_thresh = tea_cache
                trans.teacache_start_step =  max(math.ceil(tea_cache_start_step_perc*num_inference_steps/100),2)
                trans.previous_residual_uncond = None
                trans.previous_modulated_input_uncond = None                
                trans.previous_residual_cond = None
                trans.previous_modulated_input_cond= None                

                trans.teacache_cache_device = "cuda" if profile==3 or profile==1 else "cpu"                                 

            video_no += 1
            status = f"Video {video_no}/{total_video}"
            progress(0, desc=status + " - Encoding Prompt" )   
            
            callback = build_callback(state, trans, progress, status, num_inference_steps)


            gc.collect()
            torch.cuda.empty_cache()
            wan_model._interrupt = False
            try:
                if use_image2video:
                    samples = wan_model.generate(
                        prompt,
                        image_to_continue[video_no-1],
                        frame_num=(video_length // 4)* 4 + 1,
                        max_area=MAX_AREA_CONFIGS[resolution], 
                        shift=flow_shift,
                        sampling_steps=num_inference_steps,
                        guide_scale=guidance_scale,
                        n_prompt=negative_prompt,
                        seed=seed,
                        offload_model=False,
                        callback=callback,
                        enable_RIFLEx = enable_RIFLEx,
                        VAE_tile_size = VAE_tile_size
                    )

                else:
                    samples = wan_model.generate(
                        prompt,
                        frame_num=(video_length // 4)* 4 + 1,
                        size=(width, height),
                        shift=flow_shift,
                        sampling_steps=num_inference_steps,
                        guide_scale=guidance_scale,
                        n_prompt=negative_prompt,
                        seed=seed,
                        offload_model=False,
                        callback=callback,
                        enable_RIFLEx = enable_RIFLEx,
                        VAE_tile_size = VAE_tile_size
                    )
            except Exception as e:
                gen_in_progress = False
                if temp_filename!= None and  os.path.isfile(temp_filename):
                    os.remove(temp_filename)
                offload.last_offload_obj.unload_all()
                # if compile:
                #     cache_size = torch._dynamo.config.cache_size_limit                                      
                #     torch.compiler.reset()
                #     torch._dynamo.config.cache_size_limit = cache_size

                gc.collect()
                torch.cuda.empty_cache()
                s = str(e)
                keyword_list = ["vram", "VRAM", "memory", "triton", "cuda", "allocat"]
                VRAM_crash= False
                if any( keyword in s for keyword in keyword_list):
                    VRAM_crash = True
                else:
                    stack = traceback.extract_stack(f=None, limit=5)
                    for frame in stack:
                        if any( keyword in frame.name for keyword in keyword_list):
                            VRAM_crash = True
                            break
                if VRAM_crash:
                    raise gr.Error("The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames.")
                else:
                    raise gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")

            if trans.enable_teacache:
                trans.previous_residual_uncond = None
                trans.previous_residual_cond = None

            if samples != None:
                samples = samples.to("cpu")
            offload.last_offload_obj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            if samples == None:
                end_time = time.time()
                abort = True
                yield f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s"
            else:
                sample = samples.cpu()
                # video = rearrange(sample.cpu().numpy(), "c t h w -> t h w c")

                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                if os.name == 'nt':
                    file_name = f"{time_flag}_seed{seed}_{prompt[:50].replace('/','').strip()}.mp4".replace(':',' ').replace('\\',' ')
                else:
                    file_name = f"{time_flag}_seed{seed}_{prompt[:100].replace('/','').strip()}.mp4".replace(':',' ').replace('\\',' ')
                video_path = os.path.join(os.getcwd(), "gradio_outputs", file_name)        
                cache_video(
                    tensor=sample[None],
                    save_file=video_path,
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))

                print(f"New video saved to Path: "+video_path)
                file_list.append(video_path)
                if video_no < total_video:
                    yield  status
                else:
                    end_time = time.time()
                    yield f"Total Generation Time: {end_time-start_time:.1f}s"
            seed += 1
  
    if temp_filename!= None and  os.path.isfile(temp_filename):
        os.remove(temp_filename)
    gen_in_progress = False

new_preset_msg = "Enter a Name for a Lora Preset or Choose One Above"

def save_lset(lset_name, loras_choices, loras_mult_choices):
    global loras_presets
    
    if len(lset_name) == 0 or lset_name== new_preset_msg:
        gr.Info("Please enter a name for the preset")
        lset_choices =[("Please enter a name for a Lora Preset","")]
    else:
        lset_name = sanitize_file_name(lset_name)

        loras_choices_files = [ Path(loras[int(choice_no)]).parts[-1] for choice_no in loras_choices  ]
        lset  = {"loras" : loras_choices_files, "loras_mult" : loras_mult_choices}
        lset_name_filename = lset_name + ".lset" 
        full_lset_name_filename = os.path.join(lora_dir, lset_name_filename) 

        with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(lset))

        if lset_name in loras_presets:
            gr.Info(f"Lora Preset '{lset_name}' has been updated")
        else:
            gr.Info(f"Lora Preset '{lset_name}' has been created")
            loras_presets.append(Path(Path(lset_name_filename).parts[-1]).stem )
        lset_choices = [ ( preset, preset) for preset in loras_presets ]
        lset_choices.append( (new_preset_msg, ""))

    return gr.Dropdown(choices=lset_choices, value= lset_name)

def delete_lset(lset_name):
    global loras_presets
    lset_name_filename = os.path.join(lora_dir,  sanitize_file_name(lset_name) + ".lset" )
    if len(lset_name) > 0 and lset_name != new_preset_msg:
        if not os.path.isfile(lset_name_filename):
            raise gr.Error(f"Preset '{lset_name}' not found ")
        os.remove(lset_name_filename)
        pos = loras_presets.index(lset_name) 
        gr.Info(f"Lora Preset '{lset_name}' has been deleted")
        loras_presets.remove(lset_name)
    else:
        pos = len(loras_presets) 
        gr.Info(f"Choose a Preset to delete")

    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((new_preset_msg, ""))
    return  gr.Dropdown(choices=lset_choices, value= lset_choices[pos][1])

def apply_lset(lset_name, loras_choices, loras_mult_choices):

    if len(lset_name) == 0 or lset_name== new_preset_msg:
        gr.Info("Please choose a preset in the list or create one")
    else:
        loras_choices, loras_mult_choices= extract_preset(lset_name, loras)
        gr.Info(f"Lora Preset '{lset_name}' has been applied")

    return loras_choices, loras_mult_choices

def create_demo():
    
    default_inference_steps = 30
    default_flow_shift = get_default_flow(transformer_filename_i2v if use_image2video else transformer_filename_t2v)
    
    with gr.Blocks() as demo:
        state = gr.State({})
       
        if use_image2video:
            gr.Markdown("<div align=center><H1>Wan 2.1<SUP>GP</SUP> v1 - AI Image To Video Generator (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A> / <A HREF='https://github.com/Wan-Video/Wan2.1'>Original by Alibaba</A>)</H1></div>")
        else:
            gr.Markdown("<div align=center><H1>Wan 2.1<SUP>GP</SUP> v1 - AI Text To Video Generator (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A> / <A HREF='https://github.com/Wan-Video/Wan2.1'>Original by Alibaba</A>)</H1></div>")

        gr.Markdown("<FONT SIZE=3>With this first release of Wan 2.1GP by <B>DeepBeepMeep</B>, the VRAM requirements have been divided by more than 2 with no quality loss</FONT>")

        if use_image2video  and False:
            pass
        else:
            gr.Markdown("The VRAM requirements will depend greatly of the resolution and the duration of the video, for instance : 24 GB of VRAM (RTX 3090 / RTX 4090), the limits are as follows:")
            gr.Markdown("- 848 x 480 with a 14B model: 80 frames (5s) : 8 GB of VRAM")
            gr.Markdown("- 848 x 480 with the 1.3B model: 80 frames (5s) : 5 GB of VRAM")
            gr.Markdown("- 1280 x 720 with a 14B model: 80 frames (5s): 11 GB of VRAM")
            gr.Markdown("It is not recommmended to generate a video longer than 8s (128 frames) even if there is still some VRAM left as some artifacts may appear")
        gr.Markdown("Please note that if your turn on compilation, the first generation step of the first video generation will be slow due to the compilation. Therefore all your tests should be done with compilation turned off.")


        # css = """<STYLE>
        #         h2 { width: 100%;  text-align: center; border-bottom: 1px solid #000; line-height: 0.1em; margin: 10px 0 20px;  } 
        #         h2 span {background:#fff;  padding:0 10px; }</STYLE>"""
        # gr.HTML(css)

        header = gr.Markdown(generate_header(transformer_filename_i2v if use_image2video else transformer_filename_t2v, compile, attention_mode)  )            

        with gr.Accordion("Video Engine Configuration - click here to change it", open = False, visible= not args.lock_config):
            gr.Markdown("For the changes to be effective you will need to restart the gradio_server. Some choices below may be locked if the app has been launched by specifying a config preset.")

            with gr.Column():
                index = transformer_choices_t2v.index(transformer_filename_t2v)
                index = 0 if index ==0 else index
                transformer_t2v_choice = gr.Dropdown(
                    choices=[
                        ("WAN 2.1 1.3B Text to Video 16 bits (recommended)- the small model for fast generations with low VRAM requirements", 0),
                        ("WAN 2.1 14B Text to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 1),
                        ("WAN 2.1 14B Text to Video quantized to 8 bits (recommended) - the default engine but quantized", 2),
                    ],
                    value= index,
                    label="Transformer model for Text to Video",
                    interactive= not lock_ui_transformer,   
                    visible=not use_image2video
                 )

                index = transformer_choices_i2v.index(transformer_filename_i2v)
                index = 0 if index ==0 else index
                transformer_i2v_choice = gr.Dropdown(
                    choices=[
                        ("WAN 2.1 - 480p 14B Image to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                        ("WAN 2.1 - 480p 14B Image to Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                        ("WAN 2.1 - 720p 14B Image to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 2),
                        ("WAN 2.1 - 720p 14B Image to Video quantized to 8 bits (recommended) - the default engine but quantized", 3),
                    ],
                    value= index,
                    label="Transformer model for Image to Video",
                    interactive= not lock_ui_transformer,
                    visible = use_image2video, ###############
                 )

                index = text_encoder_choices.index(text_encoder_filename)
                index = 0 if index ==0 else index

                text_encoder_choice = gr.Dropdown(
                    choices=[
                        ("UMT5 XXL 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                        ("UMT5 XXL quantized to 8 bits - quantized text encoder, slightly worse quality but uses less RAM", 1),
                    ],
                    value= index,
                    label="Text Encoder model"
                 )
                def check(mode): 
                    if not mode in attention_modes_supported:
                        return " (NOT INSTALLED)"
                    else:
                        return ""
                attention_choice = gr.Dropdown(
                    choices=[
                        ("Auto : pick sage2 > sage > sdpa depending on what is installed", "auto"),
                        ("Scale Dot Product Attention: default, always available", "sdpa"),
                        ("Flash" + check("flash")+ ": good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                        # ("Xformers" + check("xformers")+ ": good quality - requires additional install (usually complex, may consume less VRAM to set up on Windows without WSL)", "xformers"),
                        ("Sage" + check("sage")+ ": 30% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                        ("Sage2" + check("sage2")+ ": 40% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage2"),
                    ],
                    value= attention_mode,
                    label="Attention Type",
                    interactive= not lock_ui_attention
                 )
                gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
                compile_choice = gr.Dropdown(
                    choices=[
                        ("ON: works only on Linux / WSL", "transformer"),
                        ("OFF: no other choice if you have Windows without using WSL", "" ),
                    ],
                    value= compile,
                    label="Compile Transformer (up to 50% faster and 30% more frames but requires Linux / WSL and Flash or Sage attention)",
                    interactive= not lock_ui_compile
                 )              


                vae_config_choice = gr.Dropdown(
                    choices=[
                ("Auto", 0),
                ("Disabled (faster but may require up to 22 GB of VRAM)", 1),
                ("256 x 256 : If at least 8 GB of VRAM", 2),
                ("128 x 128 : If at least 6 GB of VRAM", 3),
                    ],
                    value= vae_config,
                    label="VAE Tiling - reduce the high VRAM requirements for VAE decoding and VAE encoding (if enabled it will be slower)"
                 )

                profile_choice = gr.Dropdown(
                    choices=[
                ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for short videos a RTX 3090 / RTX 4090", 1),
                ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
                    ],
                    value= profile,
                    label="Profile (for power users only, not needed to change it)"
                 )

                default_ui_choice = gr.Dropdown(
                    choices=[
                        ("Text to Video", "t2v"),
                        ("Image to Video", "i2v"),
                    ],
                    value= default_ui,
                    label="Default mode when launching the App if not '--t2v' ot '--i2v' switch is specified when launching the server ",
                    # visible= True ############
                 )                

                msg = gr.Markdown()            
                apply_btn  = gr.Button("Apply Changes")


        with gr.Row():
            with gr.Column():
                video_to_continue = gr.Video(label= "Video to continue", visible= use_image2video and False) #######
                if args.multiple_images:  
                    image_to_continue = gr.Gallery(
                            label="Images as a starting point for new videos", type ="pil", #file_types= "image", 
                            columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, visible=use_image2video)
                else:
                    image_to_continue = gr.Image(label= "Image as a starting point for a new video", visible=use_image2video)

                if use_image2video:
                    prompt = gr.Textbox(label="Prompts (multiple prompts separated by carriage returns will generate multiple videos)", value="Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.", lines=3)
                else:
                    prompt = gr.Textbox(label="Prompts (multiple prompts separated by carriage returns will generate multiple videos)", value="A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect.", lines=3)

                    
                with gr.Row():
                    if use_image2video:
                        resolution = gr.Dropdown(
                            choices=[
                                # 720p
                                ("720p", "1280x720"),
                                ("480p", "832x480"),
                            ],
                            value="832x480",
                            label="Resolution (video will have the same height / width ratio than the original image)"
                        )

                    else:
                        resolution = gr.Dropdown(
                            choices=[
                                # 720p
                                ("1280x720 (16:9, 720p)", "1280x720"),
                                ("720x1280 (9:16, 720p)", "720x1280"), 
                                ("1024x1024 (4:3, 720p)", "1024x024"),
                                # ("832x1104 (3:4, 720p)", "832x1104"),
                                # ("960x960 (1:1, 720p)", "960x960"),
                                # 480p
                                # ("960x544 (16:9, 480p)", "960x544"),
                                ("832x480 (16:9, 480p)", "832x480"),
                                ("480x832 (9:16, 480p)", "480x832"),
                                # ("832x624 (4:3, 540p)", "832x624"), 
                                # ("624x832 (3:4, 540p)", "624x832"),
                                # ("720x720 (1:1, 540p)", "720x720"),
                            ],
                            value="832x480",
                            label="Resolution"
                        )

                with gr.Row():
                    with gr.Column():
                        video_length = gr.Slider(5, 193, value=81, step=4, label="Number of frames (16 = 1s)")
                    with gr.Column():
                        num_inference_steps = gr.Slider(1, 100, value=  default_inference_steps, step=1, label="Number of Inference Steps")

                with gr.Row():
                    max_frames = gr.Slider(1, 100, value=9, step=1, label="Number of input frames to use for Video2World prediction", visible=use_image2video and False) #########
    

                with gr.Row(visible= len(loras)>0):
                    lset_choices = [ (preset, preset) for preset in loras_presets ] + [(new_preset_msg, "")]
                    with gr.Column(scale=5):
                        lset_name = gr.Dropdown(show_label=False, allow_custom_value= True, scale=5, filterable=True, choices= lset_choices, value=default_lora_preset)
                    with gr.Column(scale=1):
                        # with gr.Column():
                        with gr.Row(height=17):
                            apply_lset_btn = gr.Button("Apply Lora Preset", size="sm", min_width= 1)
                        with gr.Row(height=17):
                            save_lset_btn = gr.Button("Save", size="sm", min_width= 1)
                            delete_lset_btn = gr.Button("Delete", size="sm", min_width= 1)


                loras_choices = gr.Dropdown(
                    choices=[
                        (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                    ],
                    value= default_loras_choices,
                    multiselect= True,
                    visible= len(loras)>0,
                    label="Activated Loras"
                )
                loras_mult_choices = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by space characters or carriage returns", value=default_loras_multis_str, visible= len(loras)>0 )

                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        seed = gr.Slider(-1, 999999999, value=-1, step=1, label="Seed (-1 for random)") 
                        repeat_generation = gr.Slider(1, 25.0, value=1.0, step=1, label="Number of Generated Video per prompt") 
                        with gr.Row():
                            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                        with gr.Row():
                            guidance_scale = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Guidance Scale", visible=True)
                            embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale", visible=False)
                            flow_shift = gr.Slider(0.0, 25.0, value= default_flow_shift, step=0.1, label="Shift Scale") 
                        tea_cache_setting = gr.Dropdown(
                            choices=[
                                ("Tea Cache Disabled", 0),
                                ("0.03 (around x1.6 speed up)", 0.03), 
                                ("0.05 (around x2 speed up)", 0.05), 
                                ("0.10 (around x3 speed up)", 0.1), 
                            ],
                            value=default_tea_cache,
                            visible=True,
                            label="Tea Cache Threshold to Skip Steps (the higher, the more steps are skipped but the lower the quality of the video (Tea Cache Consumes VRAM)"
                        )
                        tea_cache_start_step_perc = gr.Slider(2, 100, value=20, step=1, label="Tea Cache starting moment in percentage of generation (the later, the higher the quality but also the lower the speed gain)") 

                        RIFLEx_setting = gr.Dropdown(
                            choices=[
                                ("Auto (ON if Video longer than 5s)", 0),
                                ("Always ON", 1), 
                                ("Always OFF", 2), 
                            ],
                            value=0,
                            label="RIFLEx positional embedding to generate long video"
                        )

                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
            
            with gr.Column():
                gen_status = gr.Text(label="Status", interactive= False) 
                output = gr.Gallery(
                        label="Generated videos", show_label=False, elem_id="gallery"
                    , columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= False)
                generate_btn = gr.Button("Generate")
                abort_btn = gr.Button("Abort")

        save_lset_btn.click(save_lset, inputs=[lset_name, loras_choices, loras_mult_choices], outputs=[lset_name])
        delete_lset_btn.click(delete_lset, inputs=[lset_name], outputs=[lset_name])
        apply_lset_btn.click(apply_lset, inputs=[lset_name,loras_choices, loras_mult_choices], outputs=[loras_choices, loras_mult_choices])

        gen_status.change(refresh_gallery, inputs = [state], outputs = output )

        abort_btn.click(abort_generation,state,abort_btn )
        output.select(select_video, state, None )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                negative_prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale,
                repeat_generation,
                tea_cache_setting,
                tea_cache_start_step_perc,
                loras_choices,
                loras_mult_choices,
                image_to_continue,
                video_to_continue,
                max_frames,
                RIFLEx_setting,
                state
            ],
            outputs= [gen_status] #,state 

        ).then( 
            finalize_gallery,
            [state], 
            [output , abort_btn]
        )

        apply_btn.click(
                fn=apply_changes,
                inputs=[
                    state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,                            
                    profile_choice,
                    vae_config_choice,
                    default_ui_choice,
                ],
                outputs= msg
            ).then( 
            update_defaults, 
            [state, num_inference_steps,  flow_shift], 
            [num_inference_steps,  flow_shift, header]
                )

    return demo

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))

    server_name = args.server_name
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "localhost")

        
    demo = create_demo()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)

    demo.launch(server_name=server_name, server_port=server_port, share=args.share)

 
