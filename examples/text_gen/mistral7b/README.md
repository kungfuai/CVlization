## Quickstart

```
bash examples/text_gen/mistral7b/build.sh
bash examples/text_gen/mistral7b/train.sh
```

It takes roughly 2 hours on a g5 instance to train for 1000 steps.

You need to log in to huggingface and request acess to the pretrained model, to avoid the following error:

```
OSError: You are trying to access a gated repo.
```

## Reference

This is adapted from https://brev.dev/blog/fine-tuning-mistral
