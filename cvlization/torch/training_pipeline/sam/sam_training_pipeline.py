from collections import OrderedDict
from dataclasses import dataclass
import os
import numpy as np
import torch
from cvlization.data.transformed_map_dataset import TransformedMapDataset
from cvlization.specs.prediction_tasks import InstanceSegmentation
from cvlization.torch.training_pipeline.sam.sam_trainer import SamTrainer
from cvlization.torch.training_pipeline.sam.converter import (
    ConvertToSamInputs,
    get_trainable_sam_model,
)
from cvlization.torch.training_pipeline.sam.dice_loss import DiceLoss
from cvlization.torch.training_pipeline.sam.converter_util import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)
from cvlization.torch.training_pipeline.sam.viz import show_mask, show_points, show_anns

# from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def ensure_label_is_tensor(example):
    inputs, targets = example
    # return inputs, torch.tensor(targets["masks"])
    combined_mask = targets["masks"][0]
    for i, mask in enumerate(targets["masks"][1:]):
        mask_layer = mask * (1 + combined_mask.max())
        combined_mask[mask_layer > 0] = mask_layer[mask_layer > 0]

    return inputs, combined_mask.unsqueeze(0)


@dataclass
class SamTrainingPipeline:
    checkpoint_name: str = "sam_instance_seg"
    model_type: str = "vit_t"  # vit_b, vit_h
    batch_size: int = 1  # the training batch size
    n_objects_per_batch: int = (
        25  # the number of objects per batch that will be sampled
    )
    n_iterations: int = 200  # 5000  # how long we train (in iterations)
    device: str = "cuda"  # the device/GPU used for training
    n_sub_iteration: int = 8  # number of sub-iterations per iteration
    log_image_interval: int = 100  # how often we log images
    mixed_precision: bool = True  # whether to use mixed precision training
    always_output_single_mask: bool = True  # whether to always output a single mask
    use_single_point_prompt_per_object: bool = (
        True  # whether to use a single point prompt per object
    )
    use_background_point_as_single_point_prompt: bool = (
        True  # whether to use the background point as a single point prompt
    )

    def __post_init__(self):
        self.device = torch.device(self.device)

    def fit(self, dataset_builder):
        train_loader, val_loader = self._create_dataloaders(dataset_builder)
        model = self._create_model()
        trainer = self._create_trainer(model, train_loader, val_loader)
        trainer.fit(self.n_iterations)

    def _create_dataloaders(self, dataset_builder):
        train_ds = dataset_builder.training_dataset()
        val_ds = dataset_builder.validation_dataset()

        train_ds = TransformedMapDataset(
            source_dataset=train_ds, input_and_target_transform=ensure_label_is_tensor
        )
        val_ds = TransformedMapDataset(
            source_dataset=val_ds, input_and_target_transform=ensure_label_is_tensor
        )
        # TODO: Image has variable sizes. Consider resizing images and masks to a fixed size.
        # train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=None)
        # DEBUG
        train_ds._source_dataset.base_dataset.annotations = (
            train_ds.source_dataset.base_dataset.annotations[:1]
        )
        assert len(train_ds) == 1, len(train_ds)
        first_image = train_ds[0][0]
        assert (
            first_image.max() > 2
        ), f"image.max()={first_image.max()}, expecting 0~255 pixel values"
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=1, shuffle=False, collate_fn=None
        )
        # val_loader = torch.utils.data.DataLoader(
        #     val_ds, batch_size=1, shuffle=False, collate_fn=None
        # )
        val_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=1, shuffle=False, collate_fn=None
        )
        return train_loader, val_loader

    def _create_model(self):
        self.model = get_trainable_sam_model(
            model_type=self.model_type, device=self.device, freeze=["image_encoder"]
        )
        return self.model

    def _create_trainer(self, model, train_loader, val_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=10, verbose=True
        )
        # This class creates all the training data for a batch (inputs, prompts and labels).
        convert_inputs = ConvertToSamInputs()
        trainer = SamTrainer(
            name=self.checkpoint_name,
            # loss=torch.nn.BCELoss(), # not working
            loss=DiceLoss(channelwise=False),
            # loss=torch.nn.MSELoss(),
            metric=DiceLoss(),
            always_output_single_mask=self.always_output_single_mask,
            use_single_point_prompt_per_object=self.use_single_point_prompt_per_object,
            use_background_point_as_single_point_prompt=self.use_background_point_as_single_point_prompt,
            n_sub_iteration=self.n_sub_iteration,
            mixed_precision=self.mixed_precision,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            # currently we compute loss batch-wise, else we pass channelwise True
            device=self.device,
            lr_scheduler=scheduler,
            log_image_interval=self.log_image_interval,
            convert_inputs=convert_inputs,
            n_objects_per_batch=self.n_objects_per_batch,
            compile_model=False,
        )

    def _get_test_example_hard_coded(self, dataset_builder):
        # TODO: this function is for debugging.
        train_ds = dataset_builder.training_dataset()
        image = (train_ds[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        input_point = np.array([[242, 253]])
        # input_point = np.array([[143, 221]]) # neg point
        # input_point = np.array([[250, 280]])
        # input_point = np.array([[243, 221]])
        gt_mask = train_ds[0][1]["masks"][0].numpy()
        # print("image:", image.shape)
        input_label = np.array([1])
        return image, input_point, input_label, gt_mask

    def _get_test_example_with_foreground_point(self, dataset_builder):
        val_ds = dataset_builder.validation_dataset()
        image = (val_ds[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = val_ds[0][1]["masks"][0].numpy()
        # randomly pick a point based on gt_mask
        foreground_indices = np.argwhere(gt_mask > 0)
        np.random.seed(0)
        foreground_point = foreground_indices[np.random.choice(len(foreground_indices))]
        input_point = np.array([foreground_point])
        input_label = np.array([1])
        return image, input_point, input_label, gt_mask

    def _get_test_example_with_background_point(self, dataset_builder):
        val_ds = dataset_builder.validation_dataset()
        image = (val_ds[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = val_ds[0][1]["masks"][0].numpy()
        # randomly pick a point based on gt_mask
        background_indices = np.argwhere(gt_mask == 0)
        np.random.seed(0)
        background_point = background_indices[np.random.choice(len(background_indices))]
        input_point = np.array([background_point])
        input_label = np.array([1])  # still use a positive label
        return image, input_point, input_label, gt_mask

    def _plot_test_example(self, image, input_point, input_label, gt_mask):
        from cvlization.torch.training_pipeline.sam.viz import show_points
        import matplotlib.pyplot as plt

        figures = []
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis("on")
        # plt.show()
        figures.append(fig)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(gt_mask)
        plt.axis("on")
        figures.append(fig)
        return figures

    def _get_original_sam_predictors(
        self,
        points_per_side: int = 25,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9,
    ):
        # sam_checkpoint = os.path.expanduser("~/.sam_models/sam_vit_h_4b8939.pth")
        # TODO: this is hard coded for now.
        sam_checkpoint = os.path.expanduser("~/.sam_models/vit_t_mobile_sam.pth")
        # model_type = "vit_h"
        model_type = self.model_type
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        mask_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        return predictor, mask_gen

    def _get_finetuned_sam_predictors(
        self,
        points_per_side: int = 25,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9,
    ):
        # TODO: this is hard coded for now.
        sam = sam_model_registry[self.model_type]()
        checkpoint_path = "../../../../checkpoints/sam_instance_seg/best.pt"
        state = torch.load(
            checkpoint_path, map_location=self.device
        )  # , pickle_module=custom_pickle)
        model_state = state["model_state"]

        # copy the model weights from torch_em's training format
        sam_prefix = "sam."
        model_state = OrderedDict(
            [
                (k[len(sam_prefix) :] if k.startswith(sam_prefix) else k, v)
                for k, v in model_state.items()
            ]
        )
        sam.load_state_dict(model_state, strict=True)
        sam.to(self.device)

        predictor = SamPredictor(sam)
        predictor.model_type = self.model_type

        mask_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        return predictor, mask_gen

    def _plot_predicted_foregound_masks(
        self, dataset_builder, predictor, use_foreground_point_prompt: bool = True
    ):
        import matplotlib.pyplot as plt

        if use_foreground_point_prompt:
            (
                image,
                input_point,
                input_label,
                gt_mask,
            ) = self._get_test_example_with_foreground_point(dataset_builder)
        else:
            (
                image,
                input_point,
                input_label,
                gt_mask,
            ) = self._get_test_example_with_background_point(dataset_builder)
        # example_fig = self._plot_test_example(image, input_point, input_label, gt_mask)

        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # masks.shape example value: (3, 536, 559)
        assert len(masks.shape) == 3, masks.shape

        point_prompt_mask_figures = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(
                f"Mask {i+1}, Score: {score:.3f}, area: {mask.mean():.3f}", fontsize=18
            )
            plt.axis("off")
            # plt.show()
            point_prompt_mask_figures.append(fig)

        return point_prompt_mask_figures

    def _plot_predicted_anything_masks(
        self,
        dataset_builder,
        mask_gen,
        use_foreground_point_prompt: bool = True,
        alpha: float = 0.7,
    ):
        import matplotlib.pyplot as plt

        assert hasattr(mask_gen, "generate"), "mask_gen must have a generate method"

        if use_foreground_point_prompt:
            (
                image,
                input_point,
                input_label,
                gt_mask,
            ) = self._get_test_example_with_foreground_point(dataset_builder)
        else:
            (
                image,
                input_point,
                input_label,
                gt_mask,
            ) = self._get_test_example_with_background_point(dataset_builder)
        masks = mask_gen.generate(image)
        # close current figure, if any
        # plt.close()
        auto_mask_fig = plt.figure(figsize=(12, 12))
        plt.imshow(image)
        # print(masks)
        show_anns(masks, ax=plt.gca(), alpha=alpha)
        plt.axis("off")
        return [auto_mask_fig]
