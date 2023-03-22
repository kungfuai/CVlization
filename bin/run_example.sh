# --gpus all

: '
./bin/run_example.sh examples.doc_ai.huggingface.donut.doc_type.train
'

# docker-compose run --rm app python -m $@

docker-compose run --rm torch python -m cvlization.torch.training_pipeline.image_gen.ebm.uva_energy.pipeline