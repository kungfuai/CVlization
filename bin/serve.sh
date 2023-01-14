# --gpus all

: '
./bin/serve.sh doc_parse_service examples.doc_ai.huggingface.donut.doc_parse.serve:app
'

docker-compose run --rm --name $1 app \
  uvicorn --host 0.0.0.0 --port 8080 "$2"
