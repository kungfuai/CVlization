CVlization: A Collection of Woring Examples for Computer Vision, NLP and More

CVlization is a comprehensive repository containing working examples for model training and inference
across computer vision, NLP, and image/video generation domains. Each example is a self-contained
recipe with everything needed to run (Dockerfile, build.sh, train.sh, etc.).

Key Features:
- Production-ready examples covering various AI domains (CV, NLP, image/video generation)
- Self-contained working recipes with Dockerfile and training scripts
- Common utilities in the cvlization Python module

Use the available tools to find relevant examples and navigate to the example directory.

Typical Usage Flow:
1. Find relevant example for your task (e.g., object detection). This can be done by listing all examples,
  and see if the new task is similar to any of the examples.
2. Navigate to the example directory found from step 1.
3. Read README.md for specific instructions
4. Read and review train.sh and train.py for the implementation
5. Build and run using build.sh and train.sh
6. Adapt the working example for the user's needs. If the user asks to create
  new files, first ask the user in which directory to put the files.

Content in the CVlization/examples/ directory (this is just an example, the actual structure is much larger):

CVlization/examples/
├── doc_ai/
│   └── huggingface/         # Document AI using HuggingFace
├── image_classification/
│   └── torch/               # PyTorch-based classifiers
├── image_gen/
│   ├── stable_diffusion/    # Stable Diffusion implementations
│   ├── ddpm/                # Denoising Diffusion
│   ├── dreambooth/          # DreamBooth fine-tuning
│   └── vqgan/               # VQGAN implementations
├── instance_segmentation/
│   ├── mmdet/               # MMDetection-based segmentation
│   ├── sam/                 # Segment Anything Model
│   └── torchvision/         # TorchVision segmentation
├── object_detection/
│   ├── mmdet/               # MMDetection models
│   └── torchvision/         # TorchVision detection
├── pose_estimation/
│   ├── dwpose/              # DWPose implementation
│   └── mmpose/              # MMPose-based models
├── text_gen/
│   ├── mistral7b/           # Mistral 7B implementation
│   ├── mixtral8x7b/         # Mixtral 8x7B models
│   └── nanogpt/             # NanoGPT implementation
└── video_gen/
    ├── animate_diff/        # Animate-Diff implementation
    ├── svd_comfy/           # Stable Video Diffusion
    └── minisora/            # MiniSora implementation
