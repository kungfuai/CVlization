# --gpus all

: '
./bin/run_example.sh -n doc_parse -s examples.doc_ai.huggingface.donut.doc_parse.train
'

# Defaults
name="trainer"
detach=0

while getopts n:s:d flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        s) script=${OPTARG};;
        d) detach=1
    esac
done

if [ $detach -eq 1 ]
  then
    docker-compose run -d --name $name app python -u -m $script
else
    docker-compose run --rm --name $name app python -u -m $script
fi
