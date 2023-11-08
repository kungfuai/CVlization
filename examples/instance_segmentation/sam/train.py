import logging
import torch
from cvlization.lab.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.specs.prediction_tasks import InstanceSegmentation
from cvlization.torch.training_pipeline.sam.sam_trainer import SamTrainer
from cvlization.torch.training_pipeline.sam.converter import (
    ConvertToSamInputs,
    get_trainable_sam_model,
)
from cvlization.torch.training_pipeline.sam.dice_loss import DiceLoss


if __name__ == "__main__":
    checkpoint_name = "sam_hela"
    model_type: str = "vit_h"  # vit_b
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
