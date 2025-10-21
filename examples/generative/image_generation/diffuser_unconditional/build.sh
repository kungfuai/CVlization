# Run this script in the root directory of CVlization.

bin/build_torch_gpu.sh
docker build -t cvlization-diffuser-unconditional -f examples/image_gen/diffuser_unconditional/Dockerfile .