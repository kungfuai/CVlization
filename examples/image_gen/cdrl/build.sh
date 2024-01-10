# Run this script in the root directory of CVlization.

docker build -t cvlization-diffuser-gpu -f Dockerfile.diffuser-gpu .
docker build -t cvlization-cdrl -f examples/image_gen/cdrl/Dockerfile .