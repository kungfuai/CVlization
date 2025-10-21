## Running Stable Video Diffusion with only 12GB VRAM

- Check out the ComfyUI repo: `https://github.com/comfyanonymous/ComfyUI/tree/5b37270d3ad2227a30e15101a8d528ca77bd589d`
- Download `svd.safetensors` from `https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/tree/main`, and place it in `ComfyUI/models/checkpoints/`.
- Install python requirements, and start the UI with `python main.py`.
- Drag and drop `comfy_workflow_image_to_video.json` to the UI.
- Set the image, change parameters as necessary, and run the workflow with `ctrl+enter`.