import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TVF

from cvlization.dataset.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.lightning_utils import LightningModule
from cvlization.torch.torch_trainer import TorchTrainer

try:
    from torchmetrics.detection.map import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models import detection


LOGGER = logging.getLogger(__name__)


class UnifiedPanopticTorchvisionDataset(Dataset):
    """Normalize dataset targets to a shared panoptic contract.

    target keys used by the model:
    - boxes: Tensor[N,4]
    - labels: Tensor[N]
    - masks: Tensor[N,H,W]
    - semantic_map: Tensor[H,W] with class ids, optional ignore index 255
    """

    def __init__(self, base_dataset: Dataset, source: str):
        self.base_dataset = base_dataset
        self.source = source

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        target = dict(target)

        if self.source == "pennfudan":
            # Derive semantic map from instance masks and labels.
            h, w = image.shape[-2:]
            semantic_map = torch.zeros((h, w), dtype=torch.long)
            masks = target.get("masks")
            labels = target.get("labels")
            if masks is not None and labels is not None and masks.numel() > 0:
                if masks.ndim == 4:
                    masks = masks[:, 0]
                labels = labels.long()
                for i in range(masks.shape[0]):
                    semantic_map[masks[i] > 0.5] = labels[i]
            target["semantic_map"] = semantic_map
            return image, target

        if self.source == "coco_panoptic_tiny":
            # COCO panoptic seg_map is contiguous category ids with ignore=255.
            # Shift valid classes by +1 so 0 remains reserved as background.
            seg_map = target.get("seg_map")
            if seg_map is None:
                raise ValueError("Expected seg_map in COCO panoptic target.")
            seg_map = seg_map.long()
            semantic_map = torch.where(
                seg_map == 255,
                torch.full_like(seg_map, 255),
                seg_map + 1,
            )
            target["semantic_map"] = semantic_map
            return image, target

        raise ValueError(f"Unknown dataset source: {self.source}")


def combine_semantic_and_instance_outputs(
    instance_results: Dict[str, torch.Tensor],
    semantic_results: torch.Tensor,
    overlap_threshold: float = 0.5,
    stuff_area_thresh: int = 4096,
    instances_score_thresh: float = 0.5,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Simplified port of Detectron2's panoptic merge logic.

    Args:
        instance_results: dict with ``scores``, ``labels`` and ``masks`` from Mask R-CNN.
        semantic_results: tensor (H, W) with per-pixel semantic class ids.

    Returns:
        panoptic_seg: int32 tensor (H, W), segment ids.
        segments_info: list of metadata dicts.
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    if "scores" not in instance_results or "masks" not in instance_results:
        return panoptic_seg, []

    sorted_inds = torch.argsort(-instance_results["scores"])
    current_segment_id = 0
    segments_info = []

    # Mask R-CNN returns Nx1xHxW masks. Convert to NxHxW bool.
    masks = instance_results["masks"]
    if masks.ndim == 4:
        masks = masks[:, 0]
    instance_masks = masks > 0.5

    for inst_id in sorted_inds:
        score = float(instance_results["scores"][inst_id].item())
        if score < instances_score_thresh:
            break

        mask = instance_masks[inst_id]
        mask_area = int(mask.sum().item())
        if mask_area == 0:
            continue

        intersect = mask & (panoptic_seg > 0)
        intersect_area = int(intersect.sum().item())
        if intersect_area / max(mask_area, 1) > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": int(instance_results["labels"][inst_id].item()),
                "instance_id": int(inst_id.item()),
            }
        )

    # Fill remaining area with "stuff" labels from semantic map.
    for semantic_label in torch.unique(semantic_results).cpu().tolist():
        if semantic_label == 0:  # treat 0 as background/thing-reserved id
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = int(mask.sum().item())
        if mask_area < stuff_area_thresh:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": int(semantic_label),
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info


class TorchvisionPanopticLitModel(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        log_detailed_metrics: bool = False,
        log_images_every_n_epochs: int = 0,
        sem_ignore_index: int = 255,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.log_detailed_metrics = log_detailed_metrics
        self.log_images_every_n_epochs = max(0, int(log_images_every_n_epochs))
        self.sem_ignore_index = sem_ignore_index

        # Keep a Torchvision model as requested; use mask-capable architecture.
        self.detector = detection.maskrcnn_resnet50_fpn(
            num_classes=num_classes,
            pretrained_backbone=True,
        )
        if self.freeze_backbone:
            for p in self.detector.backbone.parameters():
                p.requires_grad = False

        # Lightweight semantic head on top of FPN feature map "0".
        self.semantic_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        self.train_mAP = MeanAveragePrecision(class_metrics=True)
        self.val_mAP = MeanAveragePrecision(class_metrics=True)
        self._reset_semantic_metrics()
        self._val_image_panels: List[np.ndarray] = []

    def _reset_semantic_metrics(self):
        self._train_sem_intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self._train_sem_union = torch.zeros(self.num_classes, dtype=torch.float64)
        self._train_sem_correct = torch.tensor(0.0, dtype=torch.float64)
        self._train_sem_total = torch.tensor(0.0, dtype=torch.float64)

        self._val_sem_intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self._val_sem_union = torch.zeros(self.num_classes, dtype=torch.float64)
        self._val_sem_correct = torch.tensor(0.0, dtype=torch.float64)
        self._val_sem_total = torch.tensor(0.0, dtype=torch.float64)

    @staticmethod
    def _semantic_confusion_stats(
        pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = pred.long()
        target = target.long()
        valid = target != ignore_index

        intersection = torch.zeros(num_classes, dtype=torch.float64)
        union = torch.zeros(num_classes, dtype=torch.float64)
        for class_id in range(num_classes):
            pred_c = (pred == class_id) & valid
            target_c = (target == class_id) & valid
            intersection[class_id] = torch.logical_and(pred_c, target_c).sum().double().cpu()
            union[class_id] = torch.logical_or(pred_c, target_c).sum().double().cpu()

        correct = ((pred == target) & valid).sum().double().cpu()
        total = valid.sum().double().cpu()
        return intersection, union, correct, total

    @staticmethod
    def _multiclass_dice_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = 255,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid = target != ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        target_1h = F.one_hot(safe_target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        target_1h = target_1h * valid_mask
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_1h, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(target_1h, dim=dims)
        dice_per_class = (2.0 * intersection + eps) / (denom + eps)
        return 1.0 - dice_per_class.mean()

    def forward(self, images):
        # TorchTrainer runs a preflight forward(inputs) before fit().
        if isinstance(images, tuple):
            images = list(images)
        images = [img.float() for img in images]
        return self.detector(images)

    def configure_optimizers(self):
        params = [p for p in list(self.detector.parameters()) + list(self.semantic_head.parameters()) if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    @staticmethod
    def _normalize_batch(batch: Tuple[Sequence[torch.Tensor], Sequence[Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
        images, targets = batch
        images = [img.float() for img in images]
        norm_targets = []
        for t in targets:
            t = dict(t)
            if "labels" in t and t["labels"].ndim == 2:
                t["labels"] = t["labels"][:, 0]
            norm_targets.append(t)
        return images, norm_targets

    @staticmethod
    def _build_semantic_targets(targets: List[Dict], out_h: int, out_w: int) -> torch.Tensor:
        sem_targets = []
        for t in targets:
            if "semantic_map" in t:
                sem = t["semantic_map"].long()
                if sem.ndim != 2:
                    raise ValueError(f"semantic_map must be HxW, got shape {tuple(sem.shape)}")
                sem = F.interpolate(
                    sem.unsqueeze(0).unsqueeze(0).float(),
                    size=(out_h, out_w),
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()
                sem_targets.append(sem)
                continue
            if "masks" not in t or t["masks"].numel() == 0:
                sem_targets.append(torch.zeros((out_h, out_w), dtype=torch.long, device=t["boxes"].device))
                continue

            masks = t["masks"].float()  # N,H,W
            labels = t["labels"].long()  # N
            if masks.ndim == 4:
                masks = masks[:, 0]

            resized_masks = F.interpolate(
                masks.unsqueeze(1),
                size=(out_h, out_w),
                mode="nearest",
            ).squeeze(1)

            sem = torch.zeros((out_h, out_w), dtype=torch.long, device=masks.device)
            for i in range(resized_masks.shape[0]):
                sem[resized_masks[i] > 0.5] = labels[i]
            sem_targets.append(sem)

        return torch.stack(sem_targets, dim=0)

    def _semantic_logits(
        self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        original_image_sizes = [img.shape[-2:] for img in images]
        image_list, _ = self.detector.transform(images, targets)
        features = self.detector.backbone(image_list.tensors)

        # p2-style FPN feature. OrderedDict key is usually "0".
        feat = features["0"] if "0" in features else next(iter(features.values()))
        sem_logits = self.semantic_head(feat)
        return sem_logits, original_image_sizes

    @staticmethod
    def _colorize_labels(labels: torch.Tensor) -> np.ndarray:
        palette = torch.tensor(
            [
                [0, 0, 0],        # background
                [255, 80, 80],    # pedestrian
                [80, 180, 255],   # spare class color
                [255, 220, 80],   # spare class color
            ],
            dtype=torch.uint8,
            device=labels.device,
        )
        labels = labels.long().clamp(min=0, max=palette.shape[0] - 1)
        rgb = palette[labels]
        return rgb.detach().cpu().numpy()

    def _should_log_val_images(self) -> bool:
        if self.log_images_every_n_epochs <= 0:
            return False
        return self.current_epoch % self.log_images_every_n_epochs == 0

    def _build_val_image_panels(
        self,
        images: List[torch.Tensor],
        sem_targets: List[torch.Tensor],
        sem_preds: List[torch.Tensor],
    ) -> List[np.ndarray]:
        panels: List[np.ndarray] = []
        n = min(2, len(images))
        for i in range(n):
            img = images[i].detach().cpu().clamp(0, 1)
            img = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            gt = self._colorize_labels(sem_targets[i])
            pred = self._colorize_labels(sem_preds[i])
            panel = np.concatenate([img, gt, pred], axis=1)
            panels.append(panel)
        return panels

    @staticmethod
    def _pad_panels_to_same_size(panels: List[np.ndarray]) -> List[np.ndarray]:
        if not panels:
            return panels
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        padded: List[np.ndarray] = []
        for p in panels:
            # Convert HWC uint8 ndarray to CHW uint8 tensor for torchvision padding.
            t = torch.from_numpy(p).permute(2, 0, 1).contiguous()
            pad_right = max_w - t.shape[2]
            pad_bottom = max_h - t.shape[1]
            t = TVF.pad(t, padding=[0, 0, pad_right, pad_bottom], fill=0)
            padded.append(t.permute(1, 2, 0).cpu().numpy())
        return padded

    def training_step(self, batch, batch_idx):
        images, targets = self._normalize_batch(batch)

        det_loss_dict = self.detector(images, targets)
        det_loss = sum(det_loss_dict.values())

        sem_logits, _ = self._semantic_logits(images, targets)
        sem_targets = self._build_semantic_targets(targets, sem_logits.shape[-2], sem_logits.shape[-1])
        sem_loss_ce = F.cross_entropy(sem_logits, sem_targets, ignore_index=self.sem_ignore_index)
        sem_loss_dice = self._multiclass_dice_loss(
            sem_logits, sem_targets, ignore_index=self.sem_ignore_index
        )
        sem_loss = sem_loss_ce + sem_loss_dice
        sem_preds = sem_logits.argmax(dim=1)
        inter, union, correct, total = self._semantic_confusion_stats(
            sem_preds, sem_targets, self.num_classes, ignore_index=self.sem_ignore_index
        )
        self._train_sem_intersection += inter
        self._train_sem_union += union
        self._train_sem_correct += correct
        self._train_sem_total += total

        total_loss = det_loss + 0.5 * sem_loss

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/loss_det", det_loss, prog_bar=True)
        self.log("train/loss_sem", sem_loss, prog_bar=True)
        self.log("train/loss_sem_ce", sem_loss_ce, prog_bar=False)
        self.log("train/loss_sem_dice", sem_loss_dice, prog_bar=False)
        if self.log_detailed_metrics:
            for k, v in det_loss_dict.items():
                self.log(f"train/{k}", v.detach(), prog_bar=False)

        # Compute train mAP from predictions in eval mode, while keeping
        # optimization behavior unchanged.
        with torch.no_grad():
            was_training = self.detector.training
            self.detector.eval()
            train_preds = self.detector(images)
            if was_training:
                self.detector.train()
        map_targets = []
        for t in targets:
            map_targets.append(
                {
                    "boxes": t["boxes"].to(self.device),
                    "labels": t["labels"].to(self.device),
                }
            )
        self.train_mAP.update(preds=train_preds, target=map_targets)
        return total_loss

    def on_train_epoch_end(self):
        train_map = self.train_mAP.to("cpu").compute()
        scalar_metrics = {
            k: (v.float().mean() if hasattr(v, "numel") and v.numel() > 1 else v)
            for k, v in train_map.items()
        }
        train_iou = torch.where(
            self._train_sem_union > 0,
            self._train_sem_intersection / self._train_sem_union,
            torch.zeros_like(self._train_sem_union),
        )
        train_miou = train_iou.mean().float()
        train_pixel_acc = (
            (self._train_sem_correct / self._train_sem_total).float()
            if self._train_sem_total.item() > 0
            else torch.tensor(0.0, dtype=torch.float32)
        )
        if self.log_detailed_metrics:
            metrics_to_log = scalar_metrics
        else:
            metrics_to_log = {
                k: v for k, v in scalar_metrics.items() if k in {"map", "map_50"}
            }
        self.log_dict({f"train/{k}": v for k, v in metrics_to_log.items()}, prog_bar=False)
        self.log("train/sem_miou", train_miou, prog_bar=False)
        self.log("train/sem_pixel_acc", train_pixel_acc, prog_bar=False)
        self.train_mAP.reset()
        self._train_sem_intersection.zero_()
        self._train_sem_union.zero_()
        self._train_sem_correct.zero_()
        self._train_sem_total.zero_()

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._val_image_panels = []
        images, targets = self._normalize_batch(batch)

        detections = self.detector(images)
        for t in targets:
            t["boxes"] = t["boxes"].to(self.device)
            t["labels"] = t["labels"].to(self.device)

        self.val_mAP.update(preds=detections, target=targets)

        # Build semantic logits and produce panoptic outputs for sanity checks.
        sem_logits, original_sizes = self._semantic_logits(images, None)
        sem_targets = self._build_semantic_targets(targets, sem_logits.shape[-2], sem_logits.shape[-1])
        sem_preds = sem_logits.argmax(dim=1)
        inter, union, correct, total = self._semantic_confusion_stats(
            sem_preds, sem_targets, self.num_classes, ignore_index=self.sem_ignore_index
        )
        self._val_sem_intersection += inter
        self._val_sem_union += union
        self._val_sem_correct += correct
        self._val_sem_total += total
        if batch_idx == 0 and self._should_log_val_images():
            vis_targets_list = []
            vis_preds_list = []
            for i, size in enumerate(original_sizes):
                vis_targets_list.append(
                    F.interpolate(
                        sem_targets[i : i + 1].unsqueeze(1).float(),
                        size=size,
                        mode="nearest",
                    ).squeeze(0).squeeze(0).long()
                )
                vis_preds_list.append(
                    F.interpolate(
                        sem_preds[i : i + 1].unsqueeze(1).float(),
                        size=size,
                        mode="nearest",
                    ).squeeze(0).squeeze(0).long()
                )
            self._val_image_panels = self._build_val_image_panels(
                images, vis_targets_list, vis_preds_list
            )
        panoptic_segment_count = 0
        for i, det in enumerate(detections):
            sem_i = sem_logits[i : i + 1]
            sem_i = F.interpolate(sem_i, size=original_sizes[i], mode="bilinear", align_corners=False)[0]
            semantic_ids = sem_i.argmax(dim=0)
            panoptic_seg, segments_info = combine_semantic_and_instance_outputs(
                det,
                semantic_ids,
                overlap_threshold=0.5,
                stuff_area_thresh=64,
                instances_score_thresh=0.3,
            )
            panoptic_segment_count += len(segments_info)

        self.log("val/panoptic_segments", float(panoptic_segment_count), prog_bar=False)

    def on_validation_epoch_end(self):
        mAP = self.val_mAP.to("cpu").compute()
        scalar_metrics = {k: (v.float().mean() if hasattr(v, "numel") and v.numel() > 1 else v) for k, v in mAP.items()}
        val_iou = torch.where(
            self._val_sem_union > 0,
            self._val_sem_intersection / self._val_sem_union,
            torch.zeros_like(self._val_sem_union),
        )
        val_miou = val_iou.mean().float()
        val_pixel_acc = (
            (self._val_sem_correct / self._val_sem_total).float()
            if self._val_sem_total.item() > 0
            else torch.tensor(0.0, dtype=torch.float32)
        )
        if self.log_detailed_metrics:
            metrics_to_log = scalar_metrics
        else:
            metrics_to_log = {
                k: v for k, v in scalar_metrics.items() if k in {"map", "map_50"}
            }
        self.log_dict({f"val/{k}": v for k, v in metrics_to_log.items()}, prog_bar=True)
        self.log("val/sem_miou", val_miou, prog_bar=True)
        self.log("val/sem_pixel_acc", val_pixel_acc, prog_bar=True)
        if self._val_image_panels and self.logger is not None:
            try:
                import wandb

                logger = self.logger
                experiment = None
                if hasattr(logger, "experiment"):
                    experiment = logger.experiment
                if experiment is not None and hasattr(experiment, "log"):
                    panels = self._pad_panels_to_same_size(self._val_image_panels)
                    experiment.log(
                        {
                            "val/images": [
                                wandb.Image(
                                    panel,
                                    caption="left: input, middle: gt_semantic, right: pred_semantic",
                                )
                                for panel in panels
                            ],
                            "trainer/global_step": int(self.global_step),
                        }
                    )
            except Exception:
                LOGGER.exception("Failed to log validation images to W&B.")
        self.val_mAP.reset()
        self._val_sem_intersection.zero_()
        self._val_sem_union.zero_()
        self._val_sem_correct.zero_()
        self._val_sem_total.zero_()
        self._val_image_panels = []


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        train_dataset, val_dataset, num_classes, sem_ignore_index = self.create_dataset()
        model = self.create_model(num_classes=num_classes, sem_ignore_index=sem_ignore_index)
        trainer = TorchTrainer(
            model=model,
            model_inputs=[],
            model_targets=[],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=4,
            val_batch_size=2,
            epochs=50,
            train_steps_per_epoch=50,
            val_steps_per_epoch=2,
            collate_method="zip",
            loss_function_included_in_model=True,
            experiment_tracker="wandb" if self.args.track else None,
            check_val_every_n_epoch=1,
        )
        trainer.run()

    def create_model(self, num_classes: int, sem_ignore_index: int):
        return TorchvisionPanopticLitModel(
            lr=0.0001,
            num_classes=num_classes,
            freeze_backbone=not self.args.unfreeze_backbone,
            log_detailed_metrics=self.args.log_detailed_metrics,
            log_images_every_n_epochs=self.args.log_images_every_n_epochs,
            sem_ignore_index=sem_ignore_index,
        )

    def create_dataset(self):
        if self.args.dataset == "pennfudan":
            dataset_builder = PennFudanPedestrianDatasetBuilder(
                flavor="torchvision",
                include_masks=True,
                label_offset=1,
            )
            train_dataset = UnifiedPanopticTorchvisionDataset(
                dataset_builder.training_dataset(), source="pennfudan"
            )
            val_dataset = UnifiedPanopticTorchvisionDataset(
                dataset_builder.validation_dataset(), source="pennfudan"
            )
            num_classes = 2  # background + pedestrian
            sem_ignore_index = 255
            return train_dataset, val_dataset, num_classes, sem_ignore_index

        if self.args.dataset == "coco_panoptic_tiny":
            from cvlization.dataset.coco_panoptic_tiny import CocoPanopticTinyDatasetBuilder

            dataset_builder = CocoPanopticTinyDatasetBuilder(
                flavor="torchvision",
                preload=False,
                label_offset=1,
            )
            train_dataset = UnifiedPanopticTorchvisionDataset(
                dataset_builder.training_dataset(), source="coco_panoptic_tiny"
            )
            val_dataset = UnifiedPanopticTorchvisionDataset(
                dataset_builder.validation_dataset(), source="coco_panoptic_tiny"
            )
            # +1 for reserved background class 0
            num_classes = len(dataset_builder.classes) + 1
            sem_ignore_index = 255
            return train_dataset, val_dataset, num_classes, sem_ignore_index

        raise ValueError(f"Unknown dataset: {self.args.dataset}")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        epilog="Torchvision panoptic-ish training (Mask R-CNN + semantic head + panoptic merge)."
    )
    parser.add_argument("--track", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pennfudan",
        choices=["pennfudan", "coco_panoptic_tiny"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze pretrained detector backbone (default: frozen).",
    )
    parser.add_argument(
        "--log-detailed-metrics",
        action="store_true",
        help="Log full metric set (default logs a compact subset).",
    )
    parser.add_argument(
        "--log-images-every-n-epochs",
        type=int,
        default=1,
        help="Log validation image panels to W&B every N epochs (0 disables).",
    )

    args = parser.parse_args()
    TrainingSession(args).run()
