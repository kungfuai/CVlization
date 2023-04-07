image=cvlization-torch-gpu
module=cvlization.torch.training_pipeline.image_gen.ebm.uva_energy.pipeline
docker run --runtime nvidia --rm \
    $image \
    python -m $module