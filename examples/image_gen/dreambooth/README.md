Training pipeline for Dreambooth

Adapted from https://github.com/replicate/dreambooth

Input:
- A directory of images

Output:
- Model weight

## Running on a GPU with 12G VRAM

Fine tune on images (`instance_data.zip`):

```
python -m examples.image_gen.dreambooth.train
```

Currently out of memory.