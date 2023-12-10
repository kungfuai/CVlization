"""
TODO: SAM model will be downloaded at ~/.sam_models. Consider mounting this directory in docker.

To use vit-t, install the following:
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

"""
import argparse
import logging
import torch
from cvlization.dataset.penn_fudan_pedestrian import PennFudanPedestrianDatasetBuilder
from cvlization.torch.training_pipeline.sam.sam_training_pipeline import (
    SamTrainingPipeline,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_with_one_example", "-d", action="store_true")
    parser.add_argument("--checkpoint_name", "-c", type=str, default="sam_instance_seg")
    parser.add_argument("--model_type", "-m", type=str, default="vit_t")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--n_objects_per_batch", "-n", type=int, default=25)
    parser.add_argument("--device", "-g", type=str, default="cuda")
    parser.add_argument("--n_iterations", "-i", type=int, default=200)
    parser.add_argument("--n_sub_iteration", "-s", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)

    dataset_builder = PennFudanPedestrianDatasetBuilder(
        flavor="torchvision",
        include_masks=True,
        label_offset=1,
        normalize_with_min_max=False,
    )

    SamTrainingPipeline(
        model_type=args.model_type,
        checkpoint_name=args.checkpoint_name,
        batch_size=args.batch_size,
        n_objects_per_batch=args.n_objects_per_batch,
        device=device,
        n_iterations=args.n_iterations,
        n_sub_iteration=args.n_sub_iteration,
        debug_with_one_example=args.debug_with_one_example,
    ).fit(dataset_builder)
