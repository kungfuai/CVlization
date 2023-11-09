"""
TODO: SAM model will be downloaded at ~/.sam_models. Consider mounting this directory in docker.

To use vit-t, install the following:
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

"""

import logging
import torch
from cvlization.data.transformed_map_dataset import TransformedMapDataset
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.specs.prediction_tasks import InstanceSegmentation
from cvlization.torch.training_pipeline.sam.sam_trainer import SamTrainer
from cvlization.torch.training_pipeline.sam.converter import (
    ConvertToSamInputs,
    get_trainable_sam_model,
)
from cvlization.torch.training_pipeline.sam.dice_loss import DiceLoss

def ensure_label_is_tensor(example):
    inputs, targets = example
    # return inputs, torch.tensor(targets["masks"])
    combined_mask = targets["masks"][0]
    for i, mask in enumerate(targets["masks"][1:]):
        mask_layer = mask * (i + 1)
        combined_mask[mask_layer > 0] = mask_layer[mask_layer > 0]

    return inputs, torch.tensor(combined_mask).unsqueeze(0)

def get_dataloaders():
    dataset_builder = PennFudanPedestrianDatasetBuilder(
        flavor="torchvision", include_masks=True, label_offset=1
    )
    train_ds = dataset_builder.training_dataset()
    val_ds = dataset_builder.validation_dataset()
    
    train_ds = TransformedMapDataset(source_dataset=train_ds, input_and_target_transform=ensure_label_is_tensor)
    val_ds = TransformedMapDataset(source_dataset=val_ds, input_and_target_transform=ensure_label_is_tensor)
    # TODO: Image has variable sizes. Consider resizing images and masks to a fixed size.
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=None)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=None)
    return train_loader, val_loader


if __name__ == "__main__":
    checkpoint_name = "sam_hela"
    model_type: str = "vit_t"  # vit_b, vit_h
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training
    n_iterations = 10000  # how long we train (in iterations)

    # Get the segment anything model, the optimizer and the LR scheduler
    model = get_trainable_sam_model(model_type=model_type, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=10, verbose=True
    )

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = ConvertToSamInputs()

    # Get the dataloaders
    train_loader, val_loader = get_dataloaders()
    trainer = SamTrainer(
        name=checkpoint_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        # currently we compute loss batch-wise, else we pass channelwise True
        loss=DiceLoss(channelwise=False),
        metric=DiceLoss(),
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False,
    )
    trainer.fit(n_iterations)
