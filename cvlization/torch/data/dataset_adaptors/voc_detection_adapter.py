import torch
import numpy as np


class VOCDetectionAdapter:
    @classmethod
    def transform(cls, img):

        img = np.moveaxis(np.array(img), -1, 0)
        img = img.astype(np.float) / 255.0
        return torch.tensor(img, dtype=torch.float)

    @classmethod
    def target_transform(clst, target):
        target_classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        annots = {"boxes": [], "labels": [], "hr_labels": []}

        for t in target["annotation"]["object"]:
            annots["boxes"].append(
                [
                    int(x)
                    for x in [
                        t["bndbox"]["xmin"],
                        t["bndbox"]["ymin"],
                        t["bndbox"]["xmax"],
                        t["bndbox"]["ymax"],
                    ]
                ]
            )
            annots["labels"].append(target_classes.index(t["name"]))
            annots["hr_labels"].append(t["name"])

        # annots["labels"] = torch.tensor(annots["labels"])
        # annots["boxes"] = torch.tensor(annots["boxes"])
        return annots["boxes"], annots["labels"]
