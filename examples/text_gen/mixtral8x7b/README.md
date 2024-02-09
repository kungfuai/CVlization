## Quickstart

```bash
bash examples/text_gen/mixtral8x7b/build.sh
bash examples/text_gen/mixtral8x7b/generate.sh
```

## How much to offload?

Here are some quick experiments using a simple prompt, "tell me a joke":

```
##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 0  # consumes 20.4 GB
# offload_per_layer = 1  # consumes 18 GB
# offload_per_layer = 2  # consumes 16.9 GB
# offload_per_layer = 3 # consumes 14.1 GB
# offload_per_layer = 4  # consumes 12.2 GB
# offload_per_layer = 5  # consumes  GB
```

## Reference

This is adapted from https://github.com/dvmazur/mixtral-offloading.