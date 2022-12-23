# --gpus all

: '
./bin/run_example.sh examples.doc_ai.huggingface.donut.doc_type.train
'

docker-compose run -d --name $1 app python -u -m $2
