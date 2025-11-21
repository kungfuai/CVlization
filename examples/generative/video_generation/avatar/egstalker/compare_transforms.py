#!/usr/bin/env python3
"""
Compare BFM and MediaPipe face tracking transform outputs.
"""
import json
import numpy as np
from pathlib import Path


def load_transforms(json_path):
    """Load transform JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_transforms(bfm_data, mp_data, name="train"):
    """Analyze and compare transform data."""
    print(f"\n{'='*80}")
    print(f"Comparison for {name} set")
    print(f"{'='*80}\n")

    # Camera parameters
    print("Camera Parameters:")
    print(f"  BFM focal length:       {bfm_data['focal_len']:.2f}")
    print(f"  MediaPipe focal length: {mp_data['focal_len']:.2f}")
    print(f"  Focal length ratio:     {bfm_data['focal_len'] / mp_data['focal_len']:.3f}x")
    print(f"\n  BFM center (cx, cy):       ({bfm_data['cx']:.1f}, {bfm_data['cy']:.1f})")
    print(f"  MediaPipe center (cx, cy): ({mp_data['cx']:.1f}, {mp_data['cy']:.1f})")

    # Frame counts
    print(f"\nFrame Counts:")
    print(f"  BFM:       {len(bfm_data['frames'])}")
    print(f"  MediaPipe: {len(mp_data['frames'])}")

    # Transform matrix analysis
    print(f"\nTransform Matrix Analysis:")

    # Sample first frame
    bfm_first = np.array(bfm_data['frames'][0]['transform_matrix'])
    mp_first = np.array(mp_data['frames'][0]['transform_matrix'])

    print(f"\n  First frame BFM transform:")
    print(f"    Translation (x, y, z): ({bfm_first[0, 3]:.4f}, {bfm_first[1, 3]:.4f}, {bfm_first[2, 3]:.4f})")

    print(f"\n  First frame MediaPipe transform:")
    print(f"    Translation (x, y, z): ({mp_first[0, 3]:.4f}, {mp_first[1, 3]:.4f}, {mp_first[2, 3]:.4f})")

    # Calculate average translation differences across all frames
    bfm_translations = []
    mp_translations = []

    for i in range(min(len(bfm_data['frames']), len(mp_data['frames']))):
        bfm_mat = np.array(bfm_data['frames'][i]['transform_matrix'])
        mp_mat = np.array(mp_data['frames'][i]['transform_matrix'])

        bfm_translations.append(bfm_mat[:3, 3])
        mp_translations.append(mp_mat[:3, 3])

    bfm_translations = np.array(bfm_translations)
    mp_translations = np.array(mp_translations)

    print(f"\n  Average translations across all frames:")
    print(f"    BFM mean (x, y, z):       ({bfm_translations[:, 0].mean():.4f}, {bfm_translations[:, 1].mean():.4f}, {bfm_translations[:, 2].mean():.4f})")
    print(f"    MediaPipe mean (x, y, z): ({mp_translations[:, 0].mean():.4f}, {mp_translations[:, 1].mean():.4f}, {mp_translations[:, 2].mean():.4f})")

    print(f"\n  Translation std deviation:")
    print(f"    BFM std (x, y, z):       ({bfm_translations[:, 0].std():.4f}, {bfm_translations[:, 1].std():.4f}, {bfm_translations[:, 2].std():.4f})")
    print(f"    MediaPipe std (x, y, z): ({mp_translations[:, 0].std():.4f}, {mp_translations[:, 1].std():.4f}, {mp_translations[:, 2].std():.4f})")

    # Z-axis (depth) comparison is particularly important
    print(f"\n  Z-axis (depth) comparison:")
    print(f"    BFM depth range:       [{bfm_translations[:, 2].min():.4f}, {bfm_translations[:, 2].max():.4f}]")
    print(f"    MediaPipe depth range: [{mp_translations[:, 2].min():.4f}, {mp_translations[:, 2].max():.4f}]")

    # Rotation analysis (extract rotation from first frame)
    def rotation_matrix_to_euler(R):
        """Extract Euler angles from rotation matrix (in degrees)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.degrees([x, y, z])

    bfm_euler = rotation_matrix_to_euler(bfm_first[:3, :3])
    mp_euler = rotation_matrix_to_euler(mp_first[:3, :3])

    print(f"\n  First frame rotation (Euler angles in degrees):")
    print(f"    BFM (pitch, yaw, roll):       ({bfm_euler[0]:.2f}, {bfm_euler[1]:.2f}, {bfm_euler[2]:.2f})")
    print(f"    MediaPipe (pitch, yaw, roll): ({mp_euler[0]:.2f}, {mp_euler[1]:.2f}, {mp_euler[2]:.2f})")


def main():
    """Main comparison function."""
    bfm_dir = Path("data/test_videos_bfm")
    mp_dir = Path("data/test_videos")

    print("\n" + "="*80)
    print("BFM vs MediaPipe Face Tracking Comparison")
    print("="*80)

    # Compare training set
    bfm_train = load_transforms(bfm_dir / "transforms_train.json")
    mp_train = load_transforms(mp_dir / "transforms_train.json")
    analyze_transforms(bfm_train, mp_train, "training")

    # Compare validation set
    bfm_val = load_transforms(bfm_dir / "transforms_val.json")
    mp_val = load_transforms(mp_dir / "transforms_val.json")
    analyze_transforms(bfm_val, mp_val, "validation")

    print("\n" + "="*80)
    print("Key Differences Summary")
    print("="*80)
    print("""
BFM (Basel Face Model):
  - Optimized focal length through fitting (614.4)
  - 3D morphable model with identity, expression, and texture parameters
  - Estimates lighting parameters
  - More computationally intensive
  - Better for photorealistic rendering with proper depth

MediaPipe:
  - Fixed focal length (540.0)
  - 478-vertex direct tracking
  - Faster processing
  - Simpler transform matrices
  - Good for real-time applications
    """)


if __name__ == "__main__":
    main()
