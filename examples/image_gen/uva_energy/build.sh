# Run this script in the root directory of CVlization.

bin/build_torch_gpu.sh
docker build -t cvlization-ebm -f examples/image_gen/uva_energy/Dockerfile .