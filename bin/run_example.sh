# --gpus all

: '
./bin/run_example.sh doc_parse examples.doc_ai.huggingface.donut.doc_parse.train
'

docker-compose run -d --name $1 app python -u -m $2
