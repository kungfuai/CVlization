from ...model_spec import ModelSpec
from ...prediction_tasks import ImageMatching
import kornia.feature as KF


class LOFTR(nn.Module):
    def __init__(self, pretrained_data="outdoor"):
        self.matcher = KF.LoFTR(pretrained=pretrained_data)
    
    def prediction_task(self) -> ModelSpec:
        return ImageMatching()

    def forward(self, inputs: list):
        image0 = inputs[0]
        image1 = inputs[1]
        loftr_data = {"image0": image0, "image1": image1}
        self.matcher(loftr_data)
        keypoints1 = loftr_data["mkpts0_f"]
        keypoints2 = loftr_data["mkpts1_f"]
        confidence = loftr_data["mconf"]
        return [keypoints1, keypoints2, confidence]
        

