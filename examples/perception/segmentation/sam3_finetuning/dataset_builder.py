#!/usr/bin/env python3
"""
Synthetic Dataset Builder for SAM3 Fine-tuning

Creates simple geometric shapes with segmentation masks in COCO format.
"""
import json
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


class DatasetBuilder:
    """Generates synthetic shape dataset in COCO format for SAM3 fine-tuning."""

    def __init__(
        self,
        output_dir: str = "data/test_shapes",
        num_train: int = 20,
        num_val: int = 5,
        image_size: Tuple[int, int] = (640, 480),
        shapes_per_image: Tuple[int, int] = (2, 5),
        clear_existing: bool = True
    ):
        """
        Initialize and create synthetic dataset.

        Args:
            output_dir: Directory to save the dataset
            num_train: Number of training images to generate
            num_val: Number of validation images to generate
            image_size: (width, height) of generated images
            shapes_per_image: (min, max) number of shapes per image
            clear_existing: Whether to clear existing dataset
        """
        self.output_dir = Path(output_dir)
        self.num_train = num_train
        self.num_val = num_val
        self.image_size = image_size
        self.shapes_per_image = shapes_per_image

        # Categories: circle, rectangle, triangle
        self.categories = [
            {"id": 0, "name": "circle", "supercategory": "shape"},
            {"id": 1, "name": "rectangle", "supercategory": "shape"},
            {"id": 2, "name": "triangle", "supercategory": "shape"}
        ]

        # Clear and create dataset
        if clear_existing and self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        print(f"\n{'='*80}")
        print(f"Creating Synthetic Shape Dataset")
        print(f"{'='*80}")
        print(f"Output: {self.output_dir}")
        print(f"Training images: {num_train}")
        print(f"Validation images: {num_val}")
        print(f"Image size: {image_size[0]}x{image_size[1]}")
        print(f"Shapes per image: {shapes_per_image[0]}-{shapes_per_image[1]}")
        print(f"{'='*80}\n")

        self._create_dataset()

    def _create_dataset(self):
        """Create train and validation splits."""
        self._create_split("train", self.num_train)
        self._create_split("valid", self.num_val)
        print(f"\n✓ Dataset created successfully at: {self.output_dir}")

    def _create_split(self, split_name: str, num_images: int):
        """Create a dataset split with images and annotations."""
        split_dir = self.output_dir / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": self.categories,
            "info": {
                "description": "Synthetic shapes dataset for SAM3 fine-tuning",
                "version": "1.0",
                "year": 2025,
                "contributor": "CVlization",
                "date_created": "2025-01-01"
            }
        }

        ann_id = 1

        for img_id in range(1, num_images + 1):
            # Generate image with random shapes
            image, annotations = self._generate_image()

            # Save image
            image_filename = f"synthetic_{img_id:03d}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)

            # Add image info
            coco_data["images"].append({
                "id": img_id,
                "file_name": image_filename,
                "width": self.image_size[0],
                "height": self.image_size[1]
            })

            # Add annotations
            for cat_id, polygon in annotations:
                # Convert polygon to RLE mask
                rle = mask_utils.frPyObjects([polygon], self.image_size[1], self.image_size[0])
                binary_mask = mask_utils.decode(rle)[..., 0]
                rle_encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
                rle_encoded['counts'] = rle_encoded['counts'].decode('utf-8')

                area = float(mask_utils.area(rle_encoded))

                # Get bounding box
                bbox = self._polygon_to_bbox(polygon)

                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": rle_encoded,
                    "iscrowd": 0
                })
                ann_id += 1

        # Save COCO JSON
        coco_json_path = split_dir / "_annotations.coco.json"
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"  ✓ {split_name}: {len(coco_data['images'])} images, "
              f"{len(coco_data['annotations'])} annotations")

    def _generate_image(self) -> Tuple[Image.Image, List[Tuple[int, List[float]]]]:
        """Generate a single image with random shapes.

        Returns:
            image: PIL Image
            annotations: List of (category_id, polygon) tuples
        """
        # Create blank image
        image = Image.new('RGB', self.image_size, color=(240, 240, 240))
        draw = ImageDraw.Draw(image)

        annotations = []
        num_shapes = np.random.randint(self.shapes_per_image[0], self.shapes_per_image[1] + 1)

        for _ in range(num_shapes):
            # Random category
            cat_id = np.random.randint(0, len(self.categories))
            cat_name = self.categories[cat_id]["name"]

            # Random color
            color = tuple(np.random.randint(50, 200, 3).tolist())

            # Random position and size
            center_x = np.random.randint(100, self.image_size[0] - 100)
            center_y = np.random.randint(100, self.image_size[1] - 100)
            size = np.random.randint(40, 120)

            # Draw shape and get polygon
            if cat_name == "circle":
                polygon = self._draw_circle(draw, center_x, center_y, size, color)
            elif cat_name == "rectangle":
                polygon = self._draw_rectangle(draw, center_x, center_y, size, color)
            else:  # triangle
                polygon = self._draw_triangle(draw, center_x, center_y, size, color)

            annotations.append((cat_id, polygon))

        return image, annotations

    def _draw_circle(self, draw: ImageDraw.Draw, cx: int, cy: int,
                     size: int, color: Tuple[int, int, int]) -> List[float]:
        """Draw a circle and return its polygon approximation."""
        radius = size // 2
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)

        # Approximate circle as polygon with 32 points
        num_points = 32
        polygon = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            polygon.extend([x, y])

        return polygon

    def _draw_rectangle(self, draw: ImageDraw.Draw, cx: int, cy: int,
                        size: int, color: Tuple[int, int, int]) -> List[float]:
        """Draw a rectangle and return its polygon."""
        width = size
        height = int(size * np.random.uniform(0.6, 1.4))

        x1 = cx - width // 2
        y1 = cy - height // 2
        x2 = cx + width // 2
        y2 = cy + height // 2

        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

        # Return as polygon (4 corners)
        polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
        return polygon

    def _draw_triangle(self, draw: ImageDraw.Draw, cx: int, cy: int,
                       size: int, color: Tuple[int, int, int]) -> List[float]:
        """Draw a triangle and return its polygon."""
        height = size
        base = int(size * 1.2)

        # Three vertices
        x1 = cx
        y1 = cy - height // 2
        x2 = cx - base // 2
        y2 = cy + height // 2
        x3 = cx + base // 2
        y3 = cy + height // 2

        points = [(x1, y1), (x2, y2), (x3, y3)]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)

        # Return as flat list
        polygon = [x1, y1, x2, y2, x3, y3]
        return polygon

    def _polygon_to_bbox(self, polygon: List[float]) -> List[float]:
        """Convert polygon to bounding box [x, y, width, height]."""
        xs = polygon[0::2]
        ys = polygon[1::2]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        return [x_min, y_min, x_max - x_min, y_max - y_min]


if __name__ == "__main__":
    # Create default dataset
    DatasetBuilder()
