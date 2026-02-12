"""SAM forward pass with gradients enabled, plus batch helpers."""

import torch


def _batch_to_device(batch, device):
    """Move image and box tensors in a SAM batch to the given device."""
    for item in batch:
        item["image"] = item["image"].to(device)
        item["boxes"] = item["boxes"].to(device)
    return batch


def sam_forward(model, batched_input, multimask_output=False):
    """SAM forward pass WITHOUT @torch.no_grad() so gradients can flow.

    The pip-installed segment_anything decorates Sam.forward with
    @torch.no_grad(), which blocks training.  This reimplements the same
    logic with gradients enabled.

    Critically, the predicted masks are **post-processed** (upsampled) to the
    original image resolution *before* being returned — matching the behaviour
    of the original Sam_LoRA vendored SAM where the loss is computed at full
    resolution rather than the decoder's native 256x256.
    """
    input_images = torch.stack(
        [model.preprocess(x["image"]) for x in batched_input], dim=0
    )
    image_embeddings = model.image_encoder(input_images)

    outputs = []
    for image_record, curr_embedding in zip(batched_input, image_embeddings):
        points = None
        if "point_coords" in image_record:
            points = (image_record["point_coords"], image_record["point_labels"])

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=image_record.get("boxes", None),
            masks=image_record.get("mask_inputs", None),
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upsample to original image resolution (matches original Sam_LoRA repo)
        masks = model.postprocess_masks(
            low_res_masks,
            input_size=image_record["image"].shape[-2:],
            original_size=image_record["original_size"],
        )
        outputs.append({"low_res_logits": masks, "iou_predictions": iou_predictions})
    return outputs
