# --gpus all

: '
./bin/run_example.sh -n doc_parse -s examples.doc_ai.huggingface.donut.doc_parse.train
'

# Defaults
name="trainer"

while getopts n:s: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        s) script=${OPTARG};;
    esac
done

docker-compose run --rm --name $name app python -u -m $script
