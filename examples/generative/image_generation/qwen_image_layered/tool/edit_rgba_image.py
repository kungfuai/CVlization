import os
import random

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import QwenImageEditPlusPipeline

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", torch_dtype=dtype
).to(device)
rmbg_model = (
    AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
    .eval()
    .to(device)
)
rmbg_transforms = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

MAX_SEED = np.iinfo(np.int32).max


def blend_with_green_bg(input_img):
    bg = Image.new("RGB", input_img.size, (30, 215, 96)).convert("RGBA")
    input_rgba = input_img.convert("RGBA")
    return Image.alpha_composite(bg, input_rgba).convert("RGB")


def infer(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    progress=gr.Progress(track_tqdm=True),
):
    negative_prompt = " "
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    image = blend_with_green_bg(image)

    edited_image = pipe(
        image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    input_images = rmbg_transforms(edited_image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = rmbg_model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(edited_image.size)
    edited_image.putalpha(mask)

    return edited_image, seed


css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("## Qwen-Image-Edit (RGBA)")
        gr.Markdown(
            "Upload an RGBA layer, provide an edit instruction, and the background "
            "will remain transparent after editing."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image", show_label=False, type="pil", image_mode="RGBA"
                )

            result = gr.Image(label="Result", show_label=False, type="pil")
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                placeholder="Describe the edit instruction",
                container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=4.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_image,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7870"))
demo.launch(server_name=server_name, server_port=server_port)
