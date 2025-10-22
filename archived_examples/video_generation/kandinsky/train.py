from diffusers import AutoPipelineForText2Image, KandinskyV22PriorPipeline
import numpy as np
import torch
from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder

# pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
#     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
# ).to("cuda")
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

def prepare_data(args):
    print(f"loading data from {args.tokens_input_file}")
    data = np.load(args.tokens_input_file).astype(np.uint16)
    vocab_size = data.max() + 5
    VIDEO_BEGIN_TOKEN = data.max() + 1
    IGNORE_TOKEN = data.max() + 2
    vae_vocab_size = data.max()
    return data, vocab_size, vae_vocab_size, VIDEO_BEGIN_TOKEN, IGNORE_TOKEN


def encode_decode():
    """
    Load an image, encode it, and decode it.
    """
    db = FlyingMNISTDatasetBuilder(resolution=224)
    # db = FlyingMNISTDatasetBuilder(resolution=256)
    ds = db.training_dataset()
    video = ds[0]["video"]
    print(video.shape)
    image = video[:3, 0:1, :, :]
    image = image.permute(1, 0, 2, 3)
    image = image.to("cuda")
    # print(dir(pipe_prior))
    # image_encoder = pipe_prior.image_encoder
    prior_image_encoder = pipe.prior_image_encoder
    
    # image = pipe_prior.process_image(image)
    print("image:", image.shape, image.min(), image.max())
    encoded = prior_image_encoder(image)["image_embeds"]
    # print("encoded:", encoded.shape)
    # prior = pipe_prior.prior
    print(dir(pipe))
    # unet = pipe.unet
    movq = pipe.movq
    movq.eval()
    encoded2 = movq.encode(image.half()).latents
    print("encoded2:", encoded2.shape)
    with torch.no_grad():
        image = movq.decode(encoded2, force_not_quantize=True)["sample"]
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    # Alternatively, with diffusion
    # decoded = pipe.decoder_pipe(encoded, negative_image_embeds=None, guidance_scale=0).images[0]
    decoded = pipe.numpy_to_pil(image)
    decoded[0].save("data/tmp/decoded.png")


encode_decode()