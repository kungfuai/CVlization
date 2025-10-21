## Quick Start

```bash
bash examples/video_gen/vace/build.sh
```

```bash
bash examples/video_gen/vace/download_models.sh
```

```bash
bash examples/video_gen/vace/example_predict.sh
```

## Examples

### Extension firstframe
```bash
bash examples/video_gen/vace/predict.sh --base wan --task frameref --mode firstframe --image "benchmarks/VACE-Benchmark/assets/examples/firstframe/ori_image_1.png" --prompt ""
```

### Repainting inpainting
```bash
bash examples/video_gen/vace/predict.sh --base wan --task inpainting --mode salientmasktrack --maskaug_mode original_expand --maskaug_ratio 0.5 --video "benchmarks/VACE-Benchmark/assets/examples/inpainting/ori_video.mp4" --prompt ""
```

### Repainting outpainting
```bash
bash examples/video_gen/vace/predict.sh --base wan --task outpainting --direction 'up,down,left,right' --expand_ratio 0.3 --video "benchmarks/VACE-Benchmark/assets/examples/outpainting/ori_video.mp4" --prompt ""
```

### Control depth
```bash
bash examples/video_gen/vace/predict.sh --base wan --task depth --video "benchmarks/VACE-Benchmark/assets/examples/depth/ori_video.mp4" --prompt ""
```

### Control flow (TODO)
```bash
bash examples/video_gen/vace/predict.sh --base wan --task flow --video "benchmarks/VACE-Benchmark/assets/examples/flow/ori_video.mp4" --prompt ""
```

### Control gray
```bash
bash examples/video_gen/vace/predict.sh --base wan --task gray --video "benchmarks/VACE-Benchmark/assets/examples/gray/ori_video.mp4" --prompt ""
```

### Control pose
```bash
bash examples/video_gen/vace/predict.sh --base wan --task pose --video "benchmarks/VACE-Benchmark/assets/examples/pose/ori_video.mp4" --prompt ""
```

### Control scribble
```bash
bash examples/video_gen/vace/predict.sh --base wan --task scribble --video "benchmarks/VACE-Benchmark/assets/examples/scribble/ori_video.mp4" --prompt ""
```

### Control layout
```bash
bash examples/video_gen/vace/predict.sh --base wan --task layout_track --mode bboxtrack --bbox '54,200,614,448' --maskaug_mode bbox_expand --maskaug_ratio 0.2 --label 'bird' --video "benchmarks/VACE-Benchmark/assets/examples/layout/ori_video.mp4" --prompt ""
```
