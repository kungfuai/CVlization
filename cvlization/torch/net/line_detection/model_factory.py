import logging
from itertools import combinations
import torch
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from scipy.ndimage import zoom
from skimage.draw import line_aa

from ...metrics.msap import msTPFP, ap


LOGGER = logging.getLogger(__name__)


class TorchLineDetectionModelFactory:
    def __init__(
        self,
        num_classes: int = 1,
        net: str = "letr",
        lightning: bool = True,
        lr: float = 0.0001,
        pretrained: bool = True,
    ):
        """
        :param num_classes: Number of classes to detect, excluding the background.
        """
        self.num_classes = num_classes
        self.net = net
        self.lightning = lightning
        # TODO: consider not putting lr here.
        self.lr = lr  # applicable for lightining model only
        self.pretrained = pretrained

    def run(self):
        model, criterion, postprocessors = create_line_detection_model(
            self.num_classes, self.net, pretrained=self.pretrained
        )
        if self.lightning:
            model = LitDetector(model, criterion, postprocessors, lr=self.lr)
        return model

    @classmethod
    def model_names(cls):
        return ["letr"]


class LitDetector(LightningModule):
    # TODO: use our own evaluator. Copy the unit tests for it.
    def __init__(self, model, criterion, postprocessors, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = criterion
        self.postprocessors = postprocessors

        # TODO: move the following states to a metric class
        self.lcnn_tp = []
        self.lcnn_fp = []
        self.lcnn_scores = []
        self.msap_threshold = 15
        self.n_gt = 0

    def forward(self, inputs, targets=None, *args, **kwargs):
        if self.training:
            outputs = self.model(inputs, *args, **kwargs)
            loss_dict = self.criterion(outputs=outputs, targets=targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
            return {"losses": losses}
        else:
            return self.model(inputs)

    def training_step(self, train_batch, batch_idx):
        # This assumes the model's forward method returns a dictionary of losses.
        loss_dict = self.forward(*train_batch)
        assert "losses" in loss_dict
        loss = sum(loss_dict.values())
        self.log("lr", self.lr, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(k[:5], v.detach(), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch
        outputs = self.forward(images)
        assert isinstance(outputs, dict), f"detections is not a dict: {type(outputs)}"
        assert "pred_lines" in outputs

        pred_logits = outputs["pred_logits"]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        postprocessors = self.postprocessors
        results = postprocessors["line"](outputs, orig_target_sizes, "prediction")
        n_query = pred_logits.shape[1]
        # assert False, f"{len(results)} results, {len(targets)} targets"

        for i, (result, target) in enumerate(zip(results, targets)):
            rst = result["lines"]
            pred_lines = rst.view(n_query, 2, 2)
            pred_lines = pred_lines.flip([-1])  # this is yxyx format

            h, w = target["orig_size"].tolist()
            # pred_lines = outputs["pred_lines"][i].reshape((-1, 2, 2))
            # pred_lines = pred_lines.flip([-1])  # this is yxyx format

            pred_lines[:, :, 0] *= 128 / h
            pred_lines[:, :, 1] *= 128 / w

            print(
                f"pred_lines max: {pred_lines[:,:,0].max()}, {pred_lines[:,:,1].max()}"
            )

            score = result["scores"].cpu().numpy()
            line = pred_lines.cpu().numpy()
            score_idx = np.argsort(-score)
            line = line[score_idx]
            score = score[score_idx]

            lcnn_line = line
            lcnn_score = score
            fgt = self._encode_targets(target)
            gt_line = fgt["lpos"][:, :, :2]
            # gt_line = target["lines"].cpu().numpy()
            # gt_line = gt_line.reshape((-1, 2, 2))
            # gt_line = gt_line[:, :, ::-1]
            gt_line[:, :, 0] = gt_line[:, :, 0] * h
            gt_line[:, :, 1] = gt_line[:, :, 1] * w
            print(
                "gt max0:",
                gt_line[:, :, 0].max(),
                "max1:",
                gt_line[:, :, 1].max(),
                "h:",
                h,
                "w:",
                w,
            )
            self.n_gt += len(gt_line)
            threshold = self.msap_threshold

            for i in range(len(lcnn_line)):
                if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                    lcnn_line = lcnn_line[:i]
                    lcnn_score = lcnn_score[:i]
                    print(f"-------- {i} detected lines and saw a duplicate")
                    break
            tp, fp = msTPFP(lcnn_line, gt_line, threshold)
            self.lcnn_tp.append(tp)
            self.lcnn_fp.append(fp)
            self.lcnn_scores.append(lcnn_score)

    def on_validation_epoch_end(self):
        lcnn_tp = np.concatenate(self.lcnn_tp)
        lcnn_fp = np.concatenate(self.lcnn_fp)
        lcnn_scores = np.concatenate(self.lcnn_scores)
        lcnn_index = np.argsort(-lcnn_scores)
        lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / self.n_gt
        lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / self.n_gt
        smap = ap(lcnn_tp, lcnn_fp)
        self.log_dict(
            {"smap": smap},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        print(
            {
                "smap": smap,
                "gt": self.n_gt,
                "lcnn_scores": lcnn_scores.shape,
                "lcnn_tp": lcnn_tp.shape,
                "lcnn_fp": lcnn_fp.shape,
                "self.lcnn_fp": len(self.lcnn_fp),
            }
        )
        # reset metrics
        self.n_gt = 0
        self.lcnn_tp = []
        self.lcnn_fp = []
        self.lcnn_scores = []

        # LOGGER.info(f"\nValidation mAP: {float(mAP['map_50'].numpy())}")
        return smap

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def _encode_targets(self, targets_for_one_example) -> dict:
        im_rescale = (512, 512)
        heatmap_scale = (128, 128)

        orig_size = targets_for_one_example["orig_size"]
        if hasattr(orig_size, "cpu"):
            orig_size = orig_size.cpu().numpy()
        lines = targets_for_one_example["lines"]
        # assert False, f"lines max: {lines.max()}"
        if hasattr(lines, "cpu"):
            lines = lines.cpu().numpy()
        lines = lines.reshape(-1, 2, 2)
        img_height = orig_size[0]
        img_width = orig_size[1]
        fy, fx = (
            heatmap_scale[1] / img_height,
            heatmap_scale[0] / img_width,
        )
        jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
        joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
        lmap = np.zeros(heatmap_scale, dtype=np.float32)

        lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
        lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
        lines = lines[:, :, ::-1]

        junc = []
        jids = {}

        def jid(jun):
            jun = tuple(jun[:2])
            if jun in jids:
                return jids[jun]
            jids[jun] = len(junc)
            junc.append(np.array(jun + (0,)))
            return len(junc) - 1

        lnid = []
        lpos, lneg = [], []
        for v0, v1 in lines:
            lnid.append((jid(v0), jid(v1)))
            lpos.append([junc[jid(v0)], junc[jid(v1)]])

            vint0, vint1 = to_int(v0), to_int(v1)
            jmap[0][vint0] = 1
            jmap[0][vint1] = 1
            rr, cc, value = line_aa(*to_int(v0), *to_int(v1))
            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        for v in junc:
            vint = to_int(v[:2])
            joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

        llmap = zoom(lmap, [0.5, 0.5])
        lineset = set([frozenset(l) for l in lnid])
        for i0, i1 in combinations(range(len(junc)), 2):
            if frozenset([i0, i1]) not in lineset:
                v0, v1 = junc[i0], junc[i1]
                vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
                rr, cc, value = line_aa(*vint0, *vint1)
                lneg.append(
                    [v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))]
                )
                # assert np.sum((v0 - v1) ** 2) > 0.01

        assert len(lneg) != 0
        lneg.sort(key=lambda l: -l[-1])

        junc = np.array(junc, dtype=np.float32)
        Lpos = np.array(lnid, dtype=np.int32)
        Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int32)
        lpos = np.array(lpos, dtype=np.float32)
        lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

        return dict(
            jmap=jmap,  # [J, H, W]
            joff=joff,  # [J, 2, H, W]
            lmap=lmap,  # [H, W]
            junc=junc,  # [Na, 3]
            Lpos=Lpos,  # [M, 2]
            Lneg=Lneg,  # [M, 2]
            lpos=lpos,  # [Np, 2, 3]   (y, x, t) for the last dim
            lneg=lneg,  # [Nn, 2, 3]
        )


def create_line_detection_model(num_classes=1, net="letr", pretrained=True):
    if num_classes > 1:
        raise ValueError(f"num_classes > 1 not supported: {num_classes}")
    if net == "letr":
        from .letr import letr
        from .letr.letr_args import LETRArgs

        args = LETRArgs()
        model, criterion, postprocessors = letr.build(args)
        # checkpoint = torch.load(
        #     "data/exp/res50_stage2_focal/checkpoints/checkpoint0024.pth",
        #     map_location="cpu",
        # )
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
            map_location="cpu",
            check_hash=True,
        )

        new_state_dict = {}
        for k in checkpoint["model"]:
            if ("class_embed" in k) or ("bbox_embed" in k) or ("query_embed" in k):
                continue
            if ("input_proj" in k) and args.layer1_num != 3:
                continue
            new_state_dict[k] = checkpoint["model"][k]

        # missing_keys, unexpected_keys = model.load_state_dict(
        #     checkpoint["model"], strict=False
        # )
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=False
        )
        # print("missing keys:", missing_keys)
    else:
        raise ValueError(
            f"Unknown network name for object detection in torchvision: {net}"
        )
    print("********************************** Model signature:")
    print(model.forward.__doc__)

    return model, criterion, postprocessors


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))
