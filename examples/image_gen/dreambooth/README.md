# Training pipeline for Dreambooth

Input:
- A directory of images

Output:
- Model weight

## Install dependencies

```
pip install -r exmaples/image_gen/dreambooth/requirements.txt
```

## Running the fine-tune job

Fine tune on images (`instance_data.zip`):

```
python -m examples.image_gen.dreambooth.train --save_sample_prompt "a happy cat"
```

- GPU requirement: at least 24GB VRAM.
- By default, train 2000 steps with a batch size of 1.

### Expected time and output

On a 3090, it took roughly 10 minutes.

You should expect the following model files organized in subdirectories:

```bash
>> du -h checkpoints/

8.0K    checkpoints/feature_extractor
235M    checkpoints/text_encoder
8.0K    checkpoints/scheduler
1.7G    checkpoints/unet
320M    checkpoints/vae
1.6M    checkpoints/tokenizer
1.9M    checkpoints/samples
2.2G    checkpoints/
```

And if you provide the `--save_sample_prompt` parameter, you can find generated images in `samples/`.

## Command line arguments

```
usage: train.py [-h] [--instance_prompt INSTANCE_PROMPT] [--class_prompt CLASS_PROMPT]
                [--instance_data INSTANCE_DATA] [--class_data CLASS_DATA]
                [--num_class_images NUM_CLASS_IMAGES] [--save_sample_prompt SAVE_SAMPLE_PROMPT]
                [--save_sample_negative_prompt SAVE_SAMPLE_NEGATIVE_PROMPT]
                [--n_save_sample N_SAVE_SAMPLE] [--save_guidance_scale SAVE_GUIDANCE_SCALE]
                [--save_infer_steps SAVE_INFER_STEPS] [--pad_tokens] [--with_prior_preservation]
                [--prior_loss_weight PRIOR_LOSS_WEIGHT] [--seed SEED] [--resolution RESOLUTION]
                [--center_crop] [--train_text_encoder] [--train_batch_size TRAIN_BATCH_SIZE]
                [--sample_batch_size SAMPLE_BATCH_SIZE] [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--gradient_checkpointing] [--learning_rate LEARNING_RATE] [--scale_lr]
                [--lr_scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--lr_warmup_steps LR_WARMUP_STEPS] [--use_8bit_adam] [--adam_beta1 ADAM_BETA1]
                [--adam_beta2 ADAM_BETA2] [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
                [--output_dir OUTPUT_DIR] [--concepts_list CONCEPTS_LIST] [--logging_dir LOGGING_DIR]
                [--log_interval LOG_INTERVAL] [--hflip] [--mixed_precision {no,fp16,fp8,bf16}]

Train Dreambooth model using example images.

optional arguments:
  -h, --help            show this help message and exit
  --instance_prompt INSTANCE_PROMPT
                        The prompt you use to describe your training images, in the format: `a
                        [identifier] [class noun]`, where the `[identifier]` should be a rare token.
                        Relatively short sequences with 1-3 letters work the best (e.g. `sks`,
                        `xjy`). `[class noun]` is a coarse class descriptor of the subject (e.g. cat,
                        dog, watch, etc.). For example, your `instance_prompt` can be: `a sks dog`,
                        or with some extra description `a photo of a sks dog`. The trained model will
                        learn to bind a unique identifier with your specific subject in the
                        `instance_data`.
  --class_prompt CLASS_PROMPT
                        The prompt or description of the coarse class of your training images, in the
                        format of `a [class noun]`, optionally with some extra description.
                        `class_prompt` is used to alleviate overfitting to your customised images
                        (the trained model should still keep the learnt prior so that it can still
                        generate different dogs when the `[identifier]` is not in the prompt).
                        Corresponding to the examples of the `instant_prompt` above, the
                        `class_prompt` can be `a dog` or `a photo of a dog`.
  --instance_data INSTANCE_DATA
                        A ZIP file containing your training images (JPG, PNG, etc. size not
                        restricted). These images contain your 'subject' that you want the trained
                        model to embed in the output domain for later generating customized scenes
                        beyond the training images. For best results, use images without noise or
                        unrelated objects in the background.
  --class_data CLASS_DATA
                        An optional ZIP file containing the training data of class images. This
                        corresponds to `class_prompt` above, also with the purpose of keeping the
                        model generalizable. By default, the pretrained stable-diffusion model will
                        generate N images (determined by the `num_class_images` you set) based on the
                        `class_prompt` provided. But to save time or to have your preferred specific
                        set of `class_data`, you can also provide them in a ZIP file.
  --num_class_images NUM_CLASS_IMAGES
                        Minimal class images for prior preservation loss. If not enough images are
                        provided in class_data, additional images will be sampled with class_prompt.
  --save_sample_prompt SAVE_SAMPLE_PROMPT
                        The prompt used to generate sample outputs to save.
  --save_sample_negative_prompt SAVE_SAMPLE_NEGATIVE_PROMPT
                        The negative prompt used to generate sample outputs to save.
  --n_save_sample N_SAVE_SAMPLE
                        The number of samples to save.
  --save_guidance_scale SAVE_GUIDANCE_SCALE
                        CFG for save sample.
  --save_infer_steps SAVE_INFER_STEPS
                        The number of inference steps for save sample.
  --pad_tokens          Flag to pad tokens to length 77.
  --with_prior_preservation
                        Flag to add prior preservation loss.
  --prior_loss_weight PRIOR_LOSS_WEIGHT
                        Weight of prior preservation loss.
  --seed SEED           A seed for reproducible training
  --resolution RESOLUTION
                        The resolution for input images. All the images in the train/validation
                        dataset will be resized to this resolution.
  --center_crop         Whether to center crop images before resizing to resolution
  --train_text_encoder  Whether to train the text encoder
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --sample_batch_size SAMPLE_BATCH_SIZE
                        Batch size (per device) for sampling images.
  --num_train_epochs NUM_TRAIN_EPOCHS
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If provided, overrides
                        num_train_epochs.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update
                        pass.
  --gradient_checkpointing
                        Whether or not to use gradient checkpointing to save memory at the expense of
                        slower backward pass.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup period) to use.
  --scale_lr            Scale the learning rate by the number of GPUs, gradient accumulation steps,
                        and batch size.
  --lr_scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
  --adam_beta1 ADAM_BETA1
                        The beta1 parameter for the Adam optimizer.
  --adam_beta2 ADAM_BETA2
                        The beta2 parameter for the Adam optimizer.
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        Weight decay to use
  --adam_epsilon ADAM_EPSILON
                        Epsilon value for the Adam optimizer
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be
                        written.
  --concepts_list CONCEPTS_LIST
                        A list of concepts to use for training.
  --logging_dir LOGGING_DIR
                        The output directory where the logs will be written.
  --log_interval LOG_INTERVAL
                        Log every n steps.
  --hflip               Whether to randomly flip images horizontally.
  --mixed_precision {no,fp16,fp8,bf16}
                        Whether or not to use mixed precision training.
```

## Reference

Adapted from https://github.com/replicate/dreambooth.