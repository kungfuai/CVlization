import numpy as np
import torch
from cvlization.torch.training_pipeline.seem.pipeline.XDecoderPipeline import XDecoderPipeline
from cvlization.torch.training_pipeline.seem.utils.arguments import load_opt_from_config_files


def main():
    opt = load_opt_from_config_files(["cvlization/torch/training_pipeline/seem//configs/seem/focalt_unicl_lang_v1.yaml"])
    prediction_pipeline = XDecoderPipeline(opt)
    print(prediction_pipeline)
    models = prediction_pipeline.initialize_model()
    print("loaded models:", list(models.keys()))
    default_model = models["default"]
    default_model.eval()
    default_model.model.panoptic_on = False
    default_model.to("cuda")
    sem_seg_head = default_model.model.sem_seg_head
    sem_seg_head.predictor.lang_encoder.get_text_embeddings(["car"], is_eval=True)
    # "image": Tensor, image in (C, H, W) format
    random_image = torch.rand((3, 224, 224)).to("cuda")
    print("random image shape:", random_image.shape)
    batched_inputs = [{"image": random_image}]
    text_prompt = "car"
    print(default_model(batched_inputs))

    # features = self.backbone(images.tensor)
    # outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')


if __name__ == "__main__":
    main()