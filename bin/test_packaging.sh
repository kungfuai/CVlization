# run this inside the container:
cd /workspace
pip install -U .
cd ..
python -c "from cvlization.torch.training_pipeline.image_gen.ebm.uva_energy.pipeline import TrainingPipeline"