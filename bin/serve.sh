# --gpus all

: '
./bin/serve.sh -n img_caption_service -f examples.doc_ai.huggingface.donut.img_caption.serve:app -n img_caption_service -p 8080
'

# Defaults
name="prediction_service"
port="8080"

while getopts n:p:f: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        p) port=${OPTARG};;
        f) fastapi=${OPTARG};;
    esac
done

docker-compose run --rm --name "$name" -p $port:$port app \
    uvicorn --host 0.0.0.0 --port $port $fastapi
