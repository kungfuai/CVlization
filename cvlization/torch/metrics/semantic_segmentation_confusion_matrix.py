import torch

# This module taken from: https://github.com/pytorch/vision/blob/main/references/segmentation/utils.py


class SemanticSegmentationConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        iu[
            iu != iu
        ] = 0  # set nan to 0, thanks to https://stackoverflow.com/questions/64751109/pytorch-when-divided-by-zero-set-the-result-value-with-0
        return acc_global, acc, iu

    # This is for multi-gpu. If you need to use it, make sure to bring in the reduce_across_processes method too.
    # def reduce_from_all_processes(self):
    #     reduce_across_processes(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        self.acc_global = acc_global.item()
        self.acc = acc.tolist()
        self.iu = iu.tolist()
        self.mean_iou = iu.mean().item()
        return (
            "global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )
