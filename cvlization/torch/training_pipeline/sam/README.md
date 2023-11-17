## Instance segmentation by fine-tuning SAM

### Overview

Segment Anything Model (SAM) is trained for interative labeling tasks, where the user prompts the model with a point, or a set of positive and negative points, or a box, or a polygon. It is able to segment generic objects (things) and areas (stuff).

Is SAM suitable for a custom instance segmentation task? How to finetune SAM to do so? The `SamTrainingPipeline` class in this directory provides an easy way to experiment with finetuning SAM for the task of instance segmentation.

From experiments on a few-shot pedestrian segmentation task, we found fine-tuning SAM did not produce a high quality customized model. Without fine-tuning, SAM tends to over-segment, and find heads, torsos, legs as well as many parts on the background. With fine-tuning, although the model sometimes can pick up more areas of the body, it is not much improvement from original SAM. More importantly, the model still picks up many parts in the background, which we find hard to force SAM to unlearn (it is the strength of SAM anyway).

Instead of using the SAM model architecture as is, some adaption is likely needed to make it more suitable for instance segmentation. Such adaptions may include:

- allow text prompts,
- adding a classification head to categorize each mask as foreground or background.

### Installation

To use vit-t, install the following:

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Fine-tune on PennFudan pedistrian dataset

```
python -m examples.instance_segmentation.sam.train
```

After training, visualize the predicted masks in `sam_example.ipynb`.

### Reference

Source code in this directory is adapted from [torch-em](https://github.com/constantinpape/torch-em/blob/main/torch_em/trainer/default_trainer.py) and [micro-sam](https://github.com/computational-cell-analytics/micro-sam).


### TODO
- Add wandb logging.