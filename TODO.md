- `TrainingPipeline` has an unfamiliar interface `assemble`, `feed_data`. Consider conforming to the convention of a heavy keras style model with `fit(dataset)`, or `create_model()`, `prepare_datasets()`, `train()`.
- (need discussion) Possibly rewrite model components (e.g. KerasImageEncoder) as framework specific models (keras.Layer, nn.Module).
    - Reason 1: keras.Model and nn.Module is well understood. We want users to easily drop in their implementations of the model components to replace the default ones in our library.
    - Reason 2: Saving of keras.Layer/Model and nn.Module is also well understood. Additional info about the custom model (e.g. which layers to extract features from) can be saved in the custom model. Our library does not need to worry about that.
    - For keras.Layer, using keras registery can be handy. When loading the trained model using our library, the user does need to install our lib so that the custom objects are available.
- Object detection models:
    - ImageEncoder to output a list of tensors at different resolutions.
    - ImageEncoder to inlude feature pyramid layers.
    - Anchor based vs. anchor free models.
    - Encoder of bounding boxes from a sparse list to a dense tensor.
- Make sure the trainer checks that the model can be saved and loaded.
- TODO: gradient accumulation, multigpu
- TODO: keras model does not work as well for cifar10?
- TODO: move experiment tracker to experiment level
- TODO: keras training: not tracked into wandb
- TODO: prediction_task: sometimes different prediction tasks are carried out sequentially
- TODO: `import cvlization` should not require tf or torch.

## Roadmap

- Image augmentation utilities, options and defaults.
    - rotation, flip, crop, resize, color augmentation
    - cutout, cutmix
- Object detection and instance segmentation.
  - Anchor based.
    - Feature pyramid on specific backbones.
    - BoxHead (regression + classification).
    - BoxEncoder. (sparse list to dense grid)
    - ROIAlign: boxes, feature maps -> cropped feature maps.
    - Segmentation head (mask head).
    - Keypoint head.
    - Use case specific heads (e. g. garage door corners, price).
  - Anchor free.
    - TBD.
- Optimizer.
   - Gradiant accumulation.
   - Tensor reshaping.
   - Schedules.
- Pre-training.
    - Autoencoder.
    - Variational AE.
    - Contrastive SSL.
- Similarity training and nearest neighbor search.
  - Hard example mining.
  - ANNOY/FAISS/Milvus.
- Image generation.
  - GAN. DCGAN.
  - Conditional GAN.
  - Diffusion. Score matching.
- Visual-textual.
    - CLIP.
    - VCGAN-CLIP.
    - GLIDE.
- Explainability.
    - Gradcam utilities (e.g. for nested keras model).
    - lucid.
- 3D object detection.
  - Additional heads: 3DBoxHead.
- Domain adaption.
    - Few-shot.
    - One shot. Meta-learning.
    - Zero shot.
    - Adversarial loss to unlearn spurious correlation.
- Neural implicit models.
- HyperModel for image networks.
