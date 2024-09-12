## Tracker players in a soccer game and generate a bird's eye view

This example is adapted from this [Roboflow example](https://github.com/roboflow/sports/tree/main).

First, download the pre-trained models and an example video:

```
cd examples/sports/soccer_game_visual_tracking
bash download_data.sh
```

(TODO: `download_data.sh` can be part of `build.sh`. Avoid having to change directory.)

and install dependences as a docker image:

```
cd <PROJECT ROOT DIRECTORY>
bash examples/sports/soccer_game_visual_tracking
```

Then, run video tracking:

```
bash examples/sports/soccer_game_visual_tracking/predict.sh
```

In this script, we are using a 30 second clip from a soccer game. The script will track the pitch and players, identify the team, goal keepers, referee, and ball, and generate a bird's eye view video. Feel free to modify the script to use a different video or to change the tracking parameters.

You will find the output video in `examples/sports/soccer_game_visual_tracking/0bfacc_0-radar.mp4`. On a machine with GeForce 3090, the `predict.sh` script takes about 20 minutes to run, with 3GB of GPU memory used.