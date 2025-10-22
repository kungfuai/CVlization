import torch
import nodes
import folder_paths
import comfy.utils
import comfy.model_management
import os
import numpy as np
from PIL import Image


class GetImageSizePlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)}}
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "get_size"
    CATEGORY = "image"

    def get_size(self, image):
        # image is in format [batch, height, width, channels]
        batch_size = image.shape[0]
        height = image.shape[1] 
        width = image.shape[2]
        return (width, height, batch_size)


class VHS_DuplicateImages:
    @classmethod  
    def INPUT_TYPES(s):
        return {"required": { "images": ("IMAGE",),
                              "multiply_by": ("INT", {"default": 1, "min": 1, "max": 1000})}}
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "count")
    FUNCTION = "duplicate_images"
    CATEGORY = "Video Helper Suite 🎥/image"

    def duplicate_images(self, images, multiply_by):
        # Duplicate each image multiply_by times
        duplicated = images.repeat(multiply_by, 1, 1, 1)
        return (duplicated, duplicated.shape[0])


class ImpactMakeImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",) },
                "optional": { "image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",), 
                             "image5": ("IMAGE",), "image6": ("IMAGE",), "image7": ("IMAGE",), "image8": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_batch"
    CATEGORY = "ImpactPack/Util"

    def make_batch(self, image1, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None):
        images = [image1]
        for img in [image2, image3, image4, image5, image6, image7, image8]:
            if img is not None:
                images.append(img)
        
        # Concatenate all images along batch dimension
        batch = torch.cat(images, dim=0)
        return (batch,)


class VHS_VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "frame_rate": ("INT", {"default": 8, "min": 1, "max": 60}),
            "loop_count": ("INT", {"default": 0, "min": 0, "max": 100}),
            "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            "format": (["image/gif", "image/webp", "video/webm", "video/h264-mp4", "video/h265-mp4"], {"default": "video/h264-mp4"}),
            "pix_fmt": (["yuv420p", "yuv444p", "rgb24"], {"default": "yuv420p"}),
            "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
            "save_metadata": ("BOOLEAN", {"default": True}),
            "pingpong": ("BOOLEAN", {"default": False}),
            "save_output": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("VHS_FILENAMES",)
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "Video Helper Suite 🎥/video"

    def combine_video(self, images, frame_rate, loop_count, filename_prefix, format, pix_fmt, crf, save_metadata, pingpong, save_output):
        # Convert images to numpy
        if images.dim() == 4:  # [batch, height, width, channels]
            frames = []
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()
                img = (img * 255).astype(np.uint8)
                frames.append(Image.fromarray(img))

            # Save as video/gif using PIL for now (simplified implementation)
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            
            if format in ["image/gif", "image/webp"]:
                ext = "gif" if format == "image/gif" else "webp"
                filename = f"{filename_prefix}_001.{ext}"
                filepath = os.path.join(output_dir, filename)
                
                # Save as animated image
                duration = int(1000 / frame_rate)  # milliseconds per frame
                frames[0].save(filepath, save_all=True, append_images=frames[1:], 
                             duration=duration, loop=loop_count)
            else:
                # For video formats, save as individual frames for now
                # In a real implementation, you'd use ffmpeg
                for i, frame in enumerate(frames):
                    filename = f"{filename_prefix}_frame_{i:04d}.png"
                    filepath = os.path.join(output_dir, filename)
                    frame.save(filepath)
                filename = f"{filename_prefix}_frames"

            return ({"filenames": [filename]},)


class MaskPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "mask": ("MASK",)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "mask_preview"
    CATEGORY = "mask"

    def mask_preview(self, mask):
        # Convert mask to viewable format
        # This is just a preview node, so we don't need to return anything
        return {}


NODE_CLASS_MAPPINGS = {
    "GetImageSize+": GetImageSizePlus,
    "VHS_DuplicateImages": VHS_DuplicateImages,
    "ImpactMakeImageBatch": ImpactMakeImageBatch,
    "VHS_VideoCombine": VHS_VideoCombine,
    "MaskPreview": MaskPreview,
} 