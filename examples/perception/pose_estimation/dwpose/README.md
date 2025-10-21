## Option 1: Using the implementation in `mimicmotion`

The code here is adapted from [MimicMotion](https://github.com/Tencent/MimicMotion), which has a simplified implementation of DWPose using pretrained ONNX models.

```
bash examples/pose_estimation/dwpose/build.sh
bash examples/pose_estimation/dwpose/predict.sh
```

When you run `predict.sh` for the first time, it will download the pretrained models to the cache folder.

Example output:

```
DWPose: 100%|██████████████████████████████████████| 100/100 [00:02<00:00, 38.48it/s]
drawing poses to a canvas: 100%|████████████████| 100/100 [00:00<00:00, 78383.55it/s]
outputs: pose: {'body': (100, 18, 2), 'face': (100, 68, 2), 'hand': (200, 21, 2), 'height': 1920, 'width': 1080} pose_img: None
example values for body pose: [[0.49417751 0.27111848]
 [0.49505182 0.33849494]
 [0.40849521 0.33800314]
 [0.38576317 0.41767457]
 [0.38226593 0.4943952 ]]
example values for face pose: [[0.45221067 0.2632497 ]
 [0.45221067 0.27013489]
 [0.45395929 0.27603647]
 [0.45570791 0.28193806]
 [0.45920514 0.28685605]]
```

## Option 2: Using Replicate's `Cog`

This did not work for me.

```
git clone https://github.com/replicate/cog-dwpose
cog build
```

You also need to download thee pose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and detection model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)), and place them under the annotator/ckpts folder.

```
cd annotator/ckpts
gdown 12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2
gdown 1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI
```

You can run the following command to get the pose estimation results.

```
cog predict -i image=@test_images/running.jpeg -i threshold=0.3
```

The output .npz file is organized as follows:

```
{
    "person_0": {
        "body": np.array of shape (18, 2),
        "face": np.array of shape (68, 2),
        "hands": np.array of shape (2, 21, 2),
    },
    "person_1": {
        "body": np.array of shape (18, 2),
        "face": np.array of shape (68, 2),
        "hands": np.array of shape (2, 21, 2),
    },
   ..
}
```



