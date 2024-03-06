"""
Adapted from https://github.com/caganselim/flying_mnist/blob/master/flying_mnist.py
"""

import argparse
import json
import numpy as np
import sys
import os
from PIL import Image
import math
import torch


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, "wb") as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


class FlyingMNISTDatasetBuilder:

    def __init__(self, opts):
        self.opts = opts

    def training_dataset(self):
        return FlyingMNISTDataset(self.opts)

    def validation_dataset(self):
        return FlyingMNISTDataset(self.opts)


class FlyingMNISTDataset:
    """
    A map-style dataset.
    """

    def __init__(
        self, opts: dict, max_frames_per_video: int = 100, max_videos: int = 10000
    ):
        self.max_videos = max_videos
        self.max_frames_per_video = max_frames_per_video
        self.opts = opts
        self.flying_mnist = FlyingMNIST(opts)

    def __getitem__(self, idx):
        """
        Returns a video of flying MNIST digits.

        Args:
            idx (int): Index of the video.

        Returns:
            np.ndarray: A video of flying MNIST digits. Shape is (T, H, W, C).
        """
        frames = []
        self.flying_mnist.init_env(idx)
        for i in range(self.max_frames_per_video):
            frame = self.flying_mnist.generate_img()
            self.flying_mnist.update_coords()
            frame = np.array(frame)
            # print(f"Frame shape: {frame.shape}")
            frames.append(frame)
        return np.array(frames)

    def __len__(self):
        if self.max_videos is None:
            return None
        assert isinstance(
            self.max_videos, int
        ), f"max_videos must be an integer, got {type(self.max_videos)} instead."
        return self.max_videos


class FlyingMNIST:

    def __init__(self, opts):

        # Save options
        self.opts = opts
        self.knob = 150

        # Load full dataset
        self.mnist = None
        self.labels = None

        # Video related variables, will be reset for each video
        self.grayscale_digits = []
        self.colored_digits = []
        self.number_of_digits = None
        self.digit_sizes = None
        self.digit_labels = None
        self.colors = None
        self.coor_list = None
        self.xlim = None
        self.ylim = None
        self.veloc = None
        self.boundary_lookup = None
        self.PALETTE = [
            0,
            0,
            0,
            128,
            0,
            0,
            0,
            128,
            0,
            128,
            128,
            0,
            0,
            0,
            128,
            128,
            0,
            128,
            0,
            128,
            128,
            128,
            128,
            128,
            64,
            0,
            0,
            191,
            0,
            0,
            64,
            128,
            0,
            191,
            128,
            0,
            64,
            0,
            128,
        ]

        self.frame_idx = 0
        self.vid_idx = 0

        # self.create_dirs()
        self.load_dataset()

    def load_dataset(self):

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source="http://yann.lecun.com/exdb/mnist/"):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
            return data

        def load_labels(filename):

            if not os.path.exists(filename):
                download(filename)

            with gzip.open(filename, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

            return data

        if self.opts.use_trn:

            self.mnist = load_mnist_images("train-images-idx3-ubyte.gz")
            self.labels = load_labels("train-labels-idx1-ubyte.gz")

        else:

            self.mnist = load_mnist_images("t10k-images-idx3-ubyte.gz")
            self.labels = load_labels("t10k-labels-idx1-ubyte.gz")

    def create_dirs(self):

        # Create VOS style dataset generation.
        os.mkdir(self.opts.target_dir)
        os.mkdir(os.path.join(self.opts.target_dir, "JPEGImages"))
        os.mkdir(os.path.join(self.opts.target_dir, "Annotations"))
        os.mkdir(os.path.join(self.opts.target_dir, "OpticalFlow"))

        for i in range(self.opts.num_videos):
            flow_dir = os.path.join(self.opts.target_dir, "OpticalFlow", f"{i:05d}")
            os.mkdir(flow_dir)

            video_dir = os.path.join(self.opts.target_dir, "JPEGImages", f"{i:05d}")
            os.mkdir(video_dir)

            seg_dir = os.path.join(self.opts.target_dir, "Annotations", f"{i:05d}")
            os.mkdir(seg_dir)

    def init_env(self, vid_idx=0):

        self.vid_idx = vid_idx
        self.frame_idx = 0

        # Randomly select digit indices
        self.number_of_digits = np.random.randint(low=1, high=self.opts.max_digits + 1)

        # Generate random digit size
        digit_sizes = np.random.randint(
            self.opts.digit_size_min, self.opts.digit_size_max, self.number_of_digits
        )

        # Select digits
        digit_indices = []
        digit_labels = np.random.choice(
            np.array(self.opts.digits), self.number_of_digits
        )

        self.digit_labels = digit_labels

        for i in range(self.number_of_digits):

            # Returns a boolean array that is true for a specific digit, filter_digits[i]
            bools = np.isin(self.labels, int(digit_labels[i]))
            idxs = np.where(bools)[0]

            # Select a digit index randomly and save it.
            digit_idx = np.random.choice(idxs)
            digit_indices.append(digit_idx)

        # Generate tuples of (x,y) proper initial positions.
        # Each coordinate (x,y) defines upper-left corner of the digit. (xmin,ymin)
        self.xlim = self.opts.canv_width - self.opts.digit_size_max - 1
        self.ylim = self.opts.canv_height - self.opts.digit_size_max - 1
        self.coor_list = np.floor(
            np.asarray(
                [
                    (np.random.rand() * self.xlim, np.random.rand() * self.ylim)
                    for _ in range(self.number_of_digits)
                ]
            )
        )

        print(self.coor_list.shape)

        # Velocity init
        direcs = np.pi * (np.random.rand(self.number_of_digits) * 2 - 1)
        speeds = np.random.randint(self.opts.max_speed, size=self.number_of_digits) + 10
        self.veloc = np.asarray(
            [
                (speed * math.cos(direc), speed * math.sin(direc))
                for direc, speed in zip(direcs, speeds)
            ]
        )

        # Select colors from a linear space
        color_basis = np.linspace(0, 1, 25)[4:]

        # Resize MNIST digits w.r.t sizes. Get size of each digit and interpolate it
        colored_digits = []
        grayscale_digits = []
        for i in range(self.number_of_digits):

            size = digit_sizes[i]
            idx = digit_indices[i]
            digit = self.mnist[idx].transpose(2, 1, 0)
            im = (
                Image.fromarray(digit[:, :, 0])
                .resize((size, size), Image.LANCZOS)
                .convert("L")
            )

            grayscale_digits.append(im)
            colored_digit = np.repeat(digit, repeats=3, axis=2) / 255.0

            if self.opts.use_coloring:
                color = np.random.choice(color_basis, 3)
            else:
                color = np.ones((3))

            # Apply colors
            colored_digit[:, :, 0] *= color[0]
            colored_digit[:, :, 1] *= color[1]
            colored_digit[:, :, 2] *= color[2]
            colored_digit = (colored_digit * 255.0).clip(0, 255).astype(np.uint8)
            im = (
                Image.fromarray(colored_digit)
                .resize((size, size), Image.LANCZOS)
                .convert("RGB")
            )
            colored_digits.append(im)

        self.colored_digits = colored_digits
        self.grayscale_digits = grayscale_digits
        self.boundary_lookup = np.zeros(self.number_of_digits, dtype=bool)

    def update_coords(self):

        # Get the next position by adding velocity
        next_coor_list = self.coor_list + self.veloc

        # Iterate over velocity and see if we hit the wall
        # If we do then change the  (change direction)
        for i, coord in enumerate(next_coor_list):

            # Calculate how many pixels can we move around a single image. => (x_lim, y_lim)
            xmin, ymin = coord[0], coord[1]

            # Check that if we hit the boundaries
            x_check = xmin < 0 or xmin > self.xlim
            y_check = ymin < 0 or ymin > self.ylim

            if y_check or x_check:

                # We hit the wall.
                # Decide to leave the scene
                if self.boundary_lookup[i]:
                    continue

                if np.random.rand() < 0.5:
                    if x_check:
                        self.veloc[i, 0] *= -1
                    if y_check:
                        self.veloc[i, 1] *= -1

                else:

                    self.boundary_lookup[i] = True

            next_coor_list = self.coor_list + self.veloc

        # Update the coordinates
        self.coor_list = next_coor_list
        self.frame_idx += 1

    def generate_img(self):

        canvas = Image.new("RGB", (self.opts.canv_width, self.opts.canv_height))

        for i in range(self.number_of_digits):

            # Create a mask
            digit_bin = self.grayscale_digits[i]
            digit_mask = np.array(digit_bin)

            digit_mask[digit_mask < self.knob] = 0
            digit_mask[digit_mask > self.knob] = 255
            digit_mask = Image.fromarray(digit_mask).convert("L")

            # Prepare coords
            coor = np.floor(self.coor_list[i, :]).astype(int)
            coor = (coor[0], coor[1])

            # Paste it
            canvas.paste(self.colored_digits[i], coor, digit_mask)

        return canvas

    def generate_seg(self):

        seg = Image.new("P", (self.opts.canv_width, self.opts.canv_height))
        seg.putpalette(self.PALETTE)

        for i in range(self.number_of_digits):

            # Create a mask
            digit_bin = self.grayscale_digits[i]
            digit_mask = np.array(digit_bin)
            label_mask = np.copy(digit_mask)

            digit_mask[digit_mask < self.knob] = 0
            digit_mask[digit_mask >= self.knob] = 255
            digit_mask = Image.fromarray(digit_mask).convert("L")

            # Prepare coords
            coor = np.floor(self.coor_list[i, :]).astype(int)
            coor = (coor[0], coor[1])

            # Seg mask
            label_mask[label_mask < self.knob] = 0
            label_mask[label_mask >= self.knob] = i + 1
            instance = Image.fromarray(label_mask).convert("P")

            # Paste it
            seg.paste(instance, coor, digit_mask)

        return seg

    def generate_flow(self):

        flow = torch.zeros(
            (1, 2, self.opts.canv_height, self.opts.canv_width), dtype=torch.float32
        )

        # Now we have the velocity and positions. Calculate flow:
        for i in range(self.number_of_digits):

            # Get image
            im = self.grayscale_digits[i]
            height, width = im.size
            digit_mask = torch.tensor(np.array(im), dtype=torch.float32)

            # Apply thresholding: convert to a mask
            digit_mask[digit_mask < self.knob] = 0
            digit_mask[digit_mask >= self.knob] = 1

            # Get coordinates
            x = int(self.coor_list[i][0])
            y = int(self.coor_list[i][1])

            # print("Digit height: ", height, " Digit width: ", width, " x: ", x, " y: ", y)

            # if x >= 473 or y >= 473 or x < -width or y < -height:
            #     continue

            if (
                x >= self.opts.canv_width
                or y >= self.opts.canv_height
                or x < -width
                or y < -height
            ):
                continue

            if x < 0:

                x_start = -x - 1
                x_end = width - 1

                digit_mask = digit_mask[:, x_start:x_end]
                # print("1 => ", x_start, " - ", x_end)

            if y < 0:

                y_start = -y - 1
                y_end = height - 1
                digit_mask = digit_mask[y_start:y_end, :]
                # print("2 => ", y_start, " - ", y_end)

            if x + width > self.opts.canv_width - 1:

                xlim = self.opts.canv_width - 1 - x
                clipped_width = xlim
                digit_mask = digit_mask[:, 0:clipped_width]

                # print("3 => ",digit_mask.shape, " x ", x, " xlim: ", xlim, " clipped_width: ", clipped_width)

            if y + height > self.opts.canv_height - 1:
                ylim = self.opts.canv_height - 1 - y
                clipped_height = ylim
                digit_mask = digit_mask[0:clipped_height, :]

                # print("4 => ", digit_mask.shape, " y ", y, " ylim: ", ylim, " clipped_height: ", clipped_height)

            # Process the limits
            x_0 = x if x > 0 else 0
            y_0 = y if y > 0 else 0
            x_1 = (
                (self.opts.canv_width - 1)
                if (x + width > self.opts.canv_width - 1)
                else x + width
            )
            y_1 = (
                (self.opts.canv_height - 1)
                if (y + height > self.opts.canv_height - 1)
                else y + height
            )

            # Update flows. Perform masking for the accurate flow.
            # First, extract mask from the pre-existing flow.
            f_x = flow[:, 0:1, y_0:y_1, x_0:x_1]
            f_y = flow[:, 1:2, y_0:y_1, x_0:x_1]

            # print(f"x: {x}, y: {y}, height: {height}, width: {width}")
            # print(f_x.shape)
            # print(f_y.shape)

            # Then apply AND operation to mask the existing flow.
            # digit_h, digit_w = f_x.shape[2], f_x.shape[3]

            # print("f_x: ", f_x.shape)
            # print("f_y: ", f_y.shape)
            # print("digit_mask: ", digit_mask.shape)

            # digit mask => 88x88, 66x66 etc
            mask_x = torch.logical_and((f_x != 0), (digit_mask > 0))
            mask_y = torch.logical_and((f_y != 0), (digit_mask > 0))

            f_x[mask_x] = 0
            f_y[mask_y] = 0

            # print("mask_x", mask_x.shape)
            # print("mask_y", mask_y.shape)
            #
            # print("A) ", flow[:, 0:1, y:y + height, x:x + width].shape)
            # print("B) ", flow[: ,1:2, y:y + height, x:x + width].shape)

            flow[:, 0:1, y_0:y_1, x_0:x_1] = f_x
            flow[:, 1:2, y_0:y_1, x_0:x_1] = f_y

            # print("A) ", flow[:, 0:1, y:y + digit_h, x:x + width].shape)
            # print("B) ", flow[: ,1:2, y:y + digit_h, x:x + width].shape)

            # print("=> ", y_0, "=> ", y_1, "=> ", x_0, "=> ", x_1)
            # print("=> " , self.veloc[i][0] , " - ", self.veloc[i][1])

            flow[:, 0:1, y_0:y_1, x_0:x_1] += (
                self.veloc[i][0] * digit_mask
            )  # x direction
            flow[:, 1:2, y_0:y_1, x_0:x_1] += (
                self.veloc[i][1] * digit_mask
            )  # y direction

        # Normalize flow here
        flow[:, 0:1, :, :] /= self.opts.canv_width
        flow[:, 1:2, :, :] /= self.opts.canv_height

        return flow

    def write_data(self):

        # Generate data
        img = self.generate_img()
        seg = self.generate_seg()
        flow = self.generate_flow()

        # Flow shape: torch.Size([1, 2, 512, 768])
        flow = flow[0].numpy()
        flow = np.swapaxes(np.swapaxes(flow, 0, 1), 1, 2)

        vid_dir = os.path.join(
            self.opts.target_dir,
            "JPEGImages",
            f"{self.vid_idx:05d}",
            f"{self.frame_idx:05d}.jpg",
        )
        seg_dir = os.path.join(
            self.opts.target_dir,
            "Annotations",
            f"{self.vid_idx:05d}",
            f"{self.frame_idx:05d}.png",
        )

        img.save(vid_dir)
        seg.save(seg_dir)

        flow_save_dir = os.path.join(
            self.opts.target_dir,
            "OpticalFlow",
            f"{self.vid_idx:05d}",
            f"{self.frame_idx:05d}.flo",
        )

        # torch.save(flow, flow_save_dir)

        writeFlowFile(flow_save_dir, flow)

    def generate(self):

        print("Generating Flying MNIST dataset...")
        for vid_idx in range(self.opts.num_videos):
            print(f"Processing video: {vid_idx}/{self.opts.num_videos}")

            self.init_env(vid_idx)

            for frame_idx in range(self.opts.num_frames):

                if np.sum(self.boundary_lookup) == len(self.boundary_lookup):
                    continue

                self.write_data()
                self.update_coords()


def prepare_parser():

    parser = argparse.ArgumentParser()

    # Input params
    parser.add_argument(
        "--canv_height", default=512, type=int, help="Canvas image height"
    )
    parser.add_argument(
        "--canv_width", default=768, type=int, help="Canvas image width"
    )
    parser.add_argument("--use_trn", default=True, help="Use MNIST train set")
    parser.add_argument(
        "--num_videos", default=100, type=int, help="Number of episodes"
    )
    parser.add_argument(
        "--num_frames", default=150, type=int, help="Number of frames in a video"
    )

    # Digit specific params
    parser.add_argument("--use_coloring", default=True, help="Apply coloring to digits")
    parser.add_argument(
        "--max_digits", default=8, type=int, help="Max number of digits"
    )
    parser.add_argument(
        "--max_speed", default=30, type=int, help="Max speed of a digit"
    )
    parser.add_argument(
        "--digit_size_min", default=50, type=int, help="Minimum digit size"
    )
    parser.add_argument(
        "--digit_size_max", default=120, type=int, help="Maximum digit size"
    )
    parser.add_argument(
        "--leaving_digits", default=True, type=str, help="Allows leaving digits"
    )
    parser.add_argument("--digits", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    parser.add_argument(
        "--target_dir", default="./trn", type=str, help="Target dir to save"
    )

    return parser


if __name__ == "__main__":
    opts = prepare_parser().parse_args()
    db = FlyingMNISTDatasetBuilder(opts)
    train_ds = db.training_dataset()
    print(f"Length of the dataset: {len(train_ds)}")
    first_video = train_ds[1]
    print(f"First item: {first_video.shape}")
    # save all frames
    import os

    os.makedirs("frames", exist_ok=True)
    frames = [f for f in first_video]
    for j, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(f"frames/frame_{j}.png")
