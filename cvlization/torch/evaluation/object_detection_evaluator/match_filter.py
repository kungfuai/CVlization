import torch
import math
from collections import Counter


class MatchFilter:
    """
    Filters a "matched" tensor
    """

    @classmethod
    def remove_double_scoring(cls, matched_detections:     torch.Tensor) ->     torch.Tensor:
        """
        If a single detection exists for more than 1 target, remove the duplicate detection entries. This method
        keeps the detection for the lowest target index
        """
        top_matches = [
            int(match)
            for match in matched_detections[3, :].tolist()
            if not math.isnan(match)
        ]
        for k, v in Counter(top_matches).items():
            if v > 1:
                indices = torch.where(matched_detections[3, :] == k)[0]
                locs = [matched_detections[1, idx] for idx in indices]
                max_idx = indices[locs.index(max(locs))]
                for idx in indices:
                    if idx != max_idx:
                        matched_detections[1, idx] = float("nan")
                        matched_detections[3, idx] = float("nan")
                        matched_detections[5, idx] = float("nan")
        return matched_detections

    @classmethod
    def remove_lower_scoring(cls, matched_detections:     torch.Tensor) ->     torch.Tensor:
        """
        If there are 2 detections corresponding to a target, remove the detection with the lowest score
        """
        for i in range(matched_detections.shape[1]):
            minimum = torch.min(matched_detections[1:3, i])
            idx = torch.where(matched_detections[1:3, i] == minimum)[0]
            if len(idx) > 1:
                idx = idx[0].unsqueeze(dim=0)
            if len(idx) > 0:
                matched_detections[int(1 + idx), i] = float("nan")
                matched_detections[3 + int(idx), i] = float("nan")
                matched_detections[5 + int(idx), i] = float("nan")
        return matched_detections

    @classmethod
    def remove_nans(cls, matched_detections:     torch.Tensor) ->     torch.Tensor:
        """Removes nans that correspond to the lowest scoring detection. If no detections are present for a particular
        target, one set of "nans" is left so that matched (matched_detections) always has a shape of (N x 4)"""
        tensor_list = []
        for i in range(matched_detections.shape[1]):
            col = matched_detections[:, i].unsqueeze(dim=0)
            if (
                sum(sum(torch.isnan(col))) < 4
            ):  # if only 1 set is nan (i.e. the entire col isn't nans)
                tensor_list.append(col[~torch.isnan(col)])
            else:
                nan_tensor = torch.tensor([float("nan")])
                new_col = col[~torch.isnan(col)]
                new_col = torch.concat([new_col, nan_tensor, nan_tensor, nan_tensor])
                tensor_list.append(new_col)
        # try:
        matched_detections = torch.stack(tensor_list, dim=1)
        # except RuntimeError as e:
        #     raise Exception(f"matched detections = {matched_detections}           ++++++++++++++ .     targets = {self.targets} .   +++++++++ .   detections = {self.detections}")
        return matched_detections
