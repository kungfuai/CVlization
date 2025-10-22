import comfy.sd
import comfy.model_sampling
import comfy.latent_formats
import nodes
import torch
import node_helpers


class ModelSamplingSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, shift, multiplier=1000):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )


class CFGZeroStar:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model):
        m = model.clone()
        
        def cfg_zero_star_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            input = args["input"]
            
            # CFG Zero Star implementation
            # This is a simplified version - the actual implementation would be more complex
            if cond_scale == 1.0:
                return cond
            else:
                return uncond + cond_scale * (cond - uncond)
        
        m.set_model_sampler_cfg_function(cfg_zero_star_function)
        return (m, )


class UNetTemporalAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "self_attn_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "cross_attn_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "temporal_attn_mult": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "output_mult": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, self_attn_mult, cross_attn_mult, temporal_attn_mult, output_mult):
        m = model.clone()
        
        def attention_multiply_patch(n, input_dict):
            # This would modify the attention weights in the model
            # Simplified implementation for demonstration
            return n
        
        # Apply the patch to the model
        m.add_object_patch("attention_multiply", {
            "self_attn_mult": self_attn_mult,
            "cross_attn_mult": cross_attn_mult, 
            "temporal_attn_mult": temporal_attn_mult,
            "output_mult": output_mult
        })
        
        return (m, )


class SkipLayerGuidanceDiT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "skip_layers": ("STRING", {"default": "9,10"}),
                              "skip_layers_cfg": ("STRING", {"default": "9,10"}),
                              "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "start": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "cfg_scale_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, skip_layers, skip_layers_cfg, scale, start, end, cfg_scale_start):
        m = model.clone()
        
        # Parse skip layers
        skip_layers_list = [int(x.strip()) for x in skip_layers.split(',') if x.strip()]
        skip_layers_cfg_list = [int(x.strip()) for x in skip_layers_cfg.split(',') if x.strip()]
        
        # Create a patch function that modifies the model's forward pass
        def skip_layer_guidance_patch(n, transformer_options):
            # This function would be called during the forward pass
            # For now, we'll implement a simple pass-through
            # In a real implementation, this would modify the skip connections
            # based on the timestep and layer indices
            return n
        
        # Use set_model_patch instead of add_object_patch to avoid attribute errors
        m.set_model_patch(skip_layer_guidance_patch, "skip_layer_guidance")
        
        # Store configuration in model options for access during inference
        if "skip_layer_guidance_config" not in m.model_options:
            m.model_options["skip_layer_guidance_config"] = {}
        
        m.model_options["skip_layer_guidance_config"].update({
            "skip_layers": skip_layers_list,
            "skip_layers_cfg": skip_layers_cfg_list,
            "scale": scale,
            "start": start,
            "end": end,
            "cfg_scale_start": cfg_scale_start
        })
        
        return (m, )


NODE_CLASS_MAPPINGS = {
    "ModelSamplingSD3": ModelSamplingSD3,
    "CFGZeroStar": CFGZeroStar,
    "UNetTemporalAttentionMultiply": UNetTemporalAttentionMultiply,
    "SkipLayerGuidanceDiT": SkipLayerGuidanceDiT,
} 