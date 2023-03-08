from PIL import Image
import requests
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
    AutoModel,
)

# TODO: add https://huggingface.co/google/owlvit-large-patch14


def get_vision_model():
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModel.from_pretrained("google/vit-base-patch16-224")
    return model, image_processor


def get_text_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return model, tokenizer


def get_vision_and_text_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        "google/vit-base-patch16-224", "bert-base-uncased"
    )
    return model, processor


def test_run_vision_text_model():
    model, processor = get_vision_and_text_model()
    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    print("image size:", images[0].size)
    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        return_loss=True,
    )
    loss, logits_per_image = outputs.loss, outputs.logits_per_image
    print(loss)
    print(logits_per_image)


def test_run_vision_model():
    model, processor = get_vision_model()
    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    print("image size:", images[0].size)
    inputs = processor(
        images=images,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(
        pixel_values=inputs.pixel_values,
        output_hidden_states=True,
    )
    print(outputs)
    print(len(outputs.hidden_states))
    print(outputs.last_hidden_state.shape)
    print(outputs.hidden_states[0].shape)
    # print(help(model))


if __name__ == "__main__":
    # contrastive training
    # test_run_vision_text_model()
    test_run_vision_model()
