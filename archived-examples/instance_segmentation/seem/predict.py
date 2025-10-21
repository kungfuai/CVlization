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
    default_model.to("cuda")
    sem_seg_head = default_model.model.sem_seg_head
    sem_seg_head.predictor.lang_encoder.get_text_embeddings(["car", "person", "background"], is_eval=True)
    default_model.eval()
    default_model.model.panoptic_on = False
    default_model.model.instance_on = True

    # Prepare inputs
    # "image": Tensor, image in (C, H, W) format
    random_image = torch.rand((3, 224, 224)).to("cuda")
    batched_inputs = [{"image": random_image, "text": ["car"]}]

    # results, image_size, extra = default_model.model.evaluate_demo(batched_inputs)
    targets = targets_grounding = queries_grounding = None
    features = default_model.model.backbone(random_image.unsqueeze(0))
    mask_features, transformer_encoder_features, multi_scale_features = sem_seg_head.pixel_decoder.forward_features(features)
    image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
    extra = {}
    
    if 'text' in batched_inputs[0]:
        gtext = sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(batched_inputs[0]['text'], name='grounding', token=False, norm=False)
        token_emb = gtext['token_emb']
        tokens = gtext['tokens']
        query_emb = token_emb[tokens['attention_mask'].bool()]
        non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)
        extra['grounding_tokens'] = query_emb[:,None]
        extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
        extra['grounding_class'] = gtext['class_emb']
        
    outputs = sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='demo')
    print("========== outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, dict):
                    for k2, v2 in vv.items():
                        print(k, k2, v2.shape)

    if False:
        print("task switch:", default_model.model.task_switch)
        default_model.to("cuda")
        print("sem_seg_head:", sem_seg_head)
        sem_seg_head.predictor.lang_encoder.get_text_embeddings(["car"], is_eval=True)
        
        print("random image shape:", random_image.shape)
        
        text_prompt = "car"
        print(default_model(batched_inputs))

        # features = self.backbone(images.tensor)
        # outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')


if __name__ == "__main__":
    main()