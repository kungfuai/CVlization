import torch
import argparse
import importlib
import os
from typing import Dict, Any

# Global variable to store the annotators path after download
_ANNOTATORS_PATH = None


def ensure_models_downloaded(base: str = "ltx"):
    """Download models lazily on first use."""
    global _ANNOTATORS_PATH
    from model_download import ensure_annotator_models, ensure_inference_models

    # Download annotator models (for preprocessing)
    _ANNOTATORS_PATH = ensure_annotator_models()

    # Download inference models
    ckpt_path = ensure_inference_models(base)
    return ckpt_path


def load_parser(module_name: str) -> argparse.ArgumentParser:
    module = importlib.import_module(module_name)
    if not hasattr(module, "get_parser"):
        raise ValueError(f"{module_name} undefined get_parser()")
    return module.get_parser()

def filter_args(args: Dict[str, Any], parser: argparse.ArgumentParser) -> Dict[str, Any]:
    known_args = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            known_args.add(action.dest)
    return {k: v for k, v in args.items() if k in known_args}

def main():

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--base", type=str, default='ltx', choices=['ltx', 'wan'])
    main_parser.add_argument("--skip_download", action="store_true",
                             help="Skip automatic model download (use local models)")
    pipeline_args, _ = main_parser.parse_known_args()

    # Lazy download models if not skipped
    if not pipeline_args.skip_download:
        print(f"Ensuring models are downloaded for base={pipeline_args.base}...")
        ckpt_path = ensure_models_downloaded(pipeline_args.base)
        print(f"Models ready. Inference checkpoint: {ckpt_path}")

        # Patch configs to use downloaded annotator paths
        from model_download import patch_config_paths
        import configs
        if _ANNOTATORS_PATH:
            configs.VACE_PREPROCCESS_CONFIGS = patch_config_paths(
                configs.VACE_PREPROCCESS_CONFIGS, _ANNOTATORS_PATH
            )

    if pipeline_args.base in ["ltx"]:
        preproccess_name, inference_name = "vace_preproccess", "vace_ltx_inference"
    else:
        preproccess_name, inference_name = "vace_preproccess", "vace_wan_inference"

    preprocess_parser = load_parser(preproccess_name)
    inference_parser = load_parser(inference_name)

    for parser in [preprocess_parser, inference_parser]:
        for action in parser._actions:
            if action.dest != "help":
                main_parser._add_action(action)

    cli_args = main_parser.parse_args()
    args_dict = vars(cli_args)

    # Set default paths if downloaded
    if not pipeline_args.skip_download:
        # For LTX models
        if pipeline_args.base == "ltx":
            if args_dict.get('ckpt_path') is None or args_dict.get('ckpt_path') == 'models/VACE-LTX-Video-0.9/ltx-video-2b-v0.9.safetensors':
                args_dict['ckpt_path'] = os.path.join(ckpt_path, 'ltx-video-2b-v0.9.safetensors')
            if args_dict.get('text_encoder_path') is None or args_dict.get('text_encoder_path') == 'models/VACE-LTX-Video-0.9':
                args_dict['text_encoder_path'] = ckpt_path
        # For Wan models
        elif pipeline_args.base == "wan":
            if args_dict.get('ckpt_path') is None or 'models/' in str(args_dict.get('ckpt_path', '')):
                args_dict['ckpt_path'] = ckpt_path

    # run preprocess
    preprocess_args = filter_args(args_dict, preprocess_parser)
    preprocesser = importlib.import_module(preproccess_name)
    preprocess_output = preprocesser.main(preprocess_args)
    print("preprocess_output:", preprocess_output)

    del preprocesser
    torch.cuda.empty_cache()

    # run inference
    inference_args = filter_args(args_dict, inference_parser)
    inference_args.update(preprocess_output)
    preprocess_output = importlib.import_module(inference_name).main(inference_args)
    print("preprocess_output:", preprocess_output)


if __name__ == "__main__":
    main()