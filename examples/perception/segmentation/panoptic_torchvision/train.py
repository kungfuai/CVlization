import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cvlization.dataset.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.lightning_utils import LightningModule
from cvlization.torch.torch_trainer import TorchTrainer

try:
    from torchmetrics.detection.map import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models import detection


LOGGER = logging.getLogger(__name__)


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
    def __init__(self, lr: float = 1e-4, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

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
        pred: torch.Tensor, target: torch.Tensor, num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = pred.long()
        target = target.long()

        intersection = torch.zeros(num_classes, dtype=torch.float64)
        union = torch.zeros(num_classes, dtype=torch.float64)
        for class_id in range(num_classes):
            pred_c = pred == class_id
            target_c = target == class_id
            intersection[class_id] = torch.logical_and(pred_c, target_c).sum().double().cpu()
            union[class_id] = torch.logical_or(pred_c, target_c).sum().double().cpu()

        correct = (pred == target).sum().double().cpu()
        total = torch.tensor(float(target.numel()), dtype=torch.float64)
        return intersection, union, correct, total

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

    def training_step(self, batch, batch_idx):
        images, targets = self._normalize_batch(batch)

        det_loss_dict = self.detector(images, targets)
        det_loss = sum(det_loss_dict.values())

        sem_logits, _ = self._semantic_logits(images, targets)
        sem_targets = self._build_semantic_targets(targets, sem_logits.shape[-2], sem_logits.shape[-1])
        sem_loss = F.cross_entropy(sem_logits, sem_targets)
        sem_preds = sem_logits.argmax(dim=1)
        inter, union, correct, total = self._semantic_confusion_stats(
            sem_preds, sem_targets, self.num_classes
        )
        self._train_sem_intersection += inter
        self._train_sem_union += union
        self._train_sem_correct += correct
        self._train_sem_total += total

        total_loss = det_loss + 0.5 * sem_loss

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/loss_det", det_loss, prog_bar=True)
        self.log("train/loss_sem", sem_loss, prog_bar=True)
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
        self.log_dict({f"train/{k}": v for k, v in scalar_metrics.items()}, prog_bar=False)
        self.log("train/sem_miou", train_miou, prog_bar=False)
        self.log("train/sem_pixel_acc", train_pixel_acc, prog_bar=False)
        self.train_mAP.reset()
        self._train_sem_intersection.zero_()
        self._train_sem_union.zero_()
        self._train_sem_correct.zero_()
        self._train_sem_total.zero_()

    def validation_step(self, batch, batch_idx):
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
            sem_preds, sem_targets, self.num_classes
        )
        self._val_sem_intersection += inter
        self._val_sem_union += union
        self._val_sem_correct += correct
        self._val_sem_total += total
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
        self.log_dict({f"val/{k}": v for k, v in scalar_metrics.items()}, prog_bar=True)
        self.log("val/sem_miou", val_miou, prog_bar=True)
        self.log("val/sem_pixel_acc", val_pixel_acc, prog_bar=True)
        self.val_mAP.reset()
        self._val_sem_intersection.zero_()
        self._val_sem_union.zero_()
        self._val_sem_correct.zero_()
        self._val_sem_total.zero_()


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        dataset_builder = self.create_dataset()
        model = self.create_model()
        trainer = TorchTrainer(
            model=model,
            model_inputs=[],
            model_targets=[],
            train_dataset=dataset_builder.training_dataset(),
            val_dataset=dataset_builder.validation_dataset(),
            train_batch_size=4,
            val_batch_size=2,
            epochs=20,
            train_steps_per_epoch=50,
            val_steps_per_epoch=2,
            collate_method="zip",
            loss_function_included_in_model=True,
            experiment_tracker=None,
            check_val_every_n_epoch=1,
        )
        trainer.run()

    def create_model(self):
        return TorchvisionPanopticLitModel(
            lr=0.0001,
            num_classes=2,
            freeze_backbone=not self.args.unfreeze_backbone,
        )

    def create_dataset(self):
        # Use masks to generate semantic targets from instance masks.
        dataset_builder = PennFudanPedestrianDatasetBuilder(
            flavor="torchvision",
            include_masks=True,
            label_offset=1,
        )
        return dataset_builder

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        epilog="Torchvision panoptic-ish training (Mask R-CNN + semantic head + panoptic merge)."
    )
    parser.add_argument("--track", action="store_true")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze pretrained detector backbone (default: frozen).",
    )

    args = parser.parse_args()
    TrainingSession(args).run()
