"""
Visual Artifact Generation for Training Data

Generates diverse visual artifacts for training video enhancement models.
Supports overlay artifacts (logos, text) and degradation artifacts
(compression, noise, blur, banding).
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, List, Optional, Dict, Any, Set, Union
import random
import math
import hashlib


class TextGenerator:
    """
    Generates diverse watermark/overlay text on the fly.

    Uses components (words, prefixes, suffixes, domains, etc.) to create
    thousands of unique text combinations deterministically based on index.
    """

    # Word components for generating diverse text
    WORDS = [
        # Common watermark words
        "Sample", "Preview", "Draft", "Proof", "Review", "Demo", "Test",
        "Watermark", "Copyright", "Protected", "Licensed", "Stock", "Photo",
        "Image", "Picture", "Media", "Video", "Clip", "Frame", "Shot",
        # Adjectives
        "Premium", "Pro", "Elite", "Best", "Top", "Great", "Cool", "Awesome",
        "Perfect", "Quality", "Creative", "Digital", "Visual", "Ultimate",
        # Nouns
        "Studio", "Gallery", "Archive", "Library", "Bank", "Hub", "Central",
        "Zone", "World", "Plus", "Max", "Source", "Market", "Shop", "Store",
    ]

    PREFIXES = ["", "My", "The", "Your", "Our", "Best", "Top", "Pro", "i", "e"]
    SUFFIXES = ["", "Pro", "Plus", "Max", "HD", "4K", "HQ", "Online", "Net", "Web"]

    DOMAINS = ["com", "net", "org", "io", "co", "tv", "media", "photo", "pics", "img"]

    YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]

    CODES = ["ABC", "XYZ", "IMG", "VID", "PIC", "REF", "ID", "CODE", "NO", "NUM"]

    SYMBOLS = ["©", "®", "™", "#", "@", "•", "★", "▶", "●"]

    # Templates for generating text
    TEMPLATES = [
        "{word}",                           # Simple: "Sample"
        "{prefix}{word}",                   # Prefix: "MyPhoto"
        "{word}{suffix}",                   # Suffix: "PhotoPro"
        "{prefix}{word}{suffix}",           # Both: "MyPhotoPro"
        "{word} {word2}",                   # Two words: "Stock Photo"
        "{word}{word2}",                    # Combined: "StockPhoto"
        "www.{word}.{domain}",              # Website: "www.sample.com"
        "{word}.{domain}",                  # Short domain: "sample.com"
        "@{word}{word2}",                   # Handle: "@stockphoto"
        "@{word}_{word2}",                  # Handle with underscore
        "{symbol} {year}",                  # Copyright: "© 2024"
        "{symbol} {word}",                  # Symbol + word: "© Sample"
        "{symbol} {year} {word}",           # Full copyright
        "{code}: {number}",                 # ID: "REF: 12345"
        "{code}{number}",                   # Code: "IMG12345"
        "#{word}{number}",                  # Hashtag: "#STOCK2024"
        "{word} {word2} {word3}",           # Three words
        "NOT FOR {word}",                   # Warning style
        "FOR {word} ONLY",                  # Restriction style
        "{word} COPY",                      # Copy style
        "{word} VERSION",                   # Version style
    ]

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def generate(self, index: int) -> str:
        """Generate a deterministic text based on index."""
        # Create deterministic random state from index
        seed = self.base_seed + index
        rng = random.Random(seed)

        # Select template
        template = rng.choice(self.TEMPLATES)

        # Fill in template
        text = template.format(
            word=rng.choice(self.WORDS),
            word2=rng.choice(self.WORDS),
            word3=rng.choice(self.WORDS),
            prefix=rng.choice(self.PREFIXES),
            suffix=rng.choice(self.SUFFIXES),
            domain=rng.choice(self.DOMAINS),
            year=rng.choice(self.YEARS),
            code=rng.choice(self.CODES),
            number=rng.randint(1000, 99999),
            symbol=rng.choice(self.SYMBOLS),
        )

        # Randomly apply case transformation
        case_choice = rng.random()
        if case_choice < 0.3:
            text = text.upper()
        elif case_choice < 0.5:
            text = text.lower()
        # else: keep original case

        return text

    def generate_batch(self, start_index: int, count: int) -> List[str]:
        """Generate a batch of texts."""
        return [self.generate(start_index + i) for i in range(count)]


class ArtifactGenerator:
    """
    Generates various visual artifacts for training data augmentation.

    Artifact types can be enabled/disabled individually via `enabled_artifacts`.
    """

    # Available artifact types
    ARTIFACT_TYPES = {
        # Overlay artifacts (logos, text) - use masks
        "corner_logo",
        "text_overlay",
        "tiled_pattern",
        "moving_logo",
        "channel_logo",
        "diagonal_text",
        # Compression artifacts
        "jpeg_compression",
        "video_compression",
        # Noise artifacts
        "gaussian_noise",
        "salt_pepper_noise",
        "film_grain",
        # Other degradations
        "color_banding",
        "blur",
    }

    # Overlay-type artifacts (these return masks)
    OVERLAY_TYPES = {
        "corner_logo", "text_overlay", "tiled_pattern",
        "moving_logo", "channel_logo", "diagonal_text"
    }

    # Degradation-type artifacts (these modify frames directly)
    DEGRADATION_TYPES = {
        "jpeg_compression", "video_compression",
        "gaussian_noise", "salt_pepper_noise", "film_grain",
        "color_banding", "blur"
    }

    # Training texts (generic watermarks, stock photo style)
    TRAIN_TEXTS = [
        # Watermark words
        "Sample", "Preview", "Draft", "Proof", "Review",
        "Watermark", "Copyright", "Protected", "Licensed",
        # Stock photo style
        "Awesome Images", "Cool Stock", "Picture Pro", "Media Hub",
        "Photo Plus", "Stock Central", "Image Bank", "Pixel Perfect",
        "Visual Arts", "Creative Stock", "Pro Photos", "Elite Images",
        # Websites
        "www.example.com", "www.stockphoto.net", "images.sample.org",
        "photos.demo.com", "stock.preview.io", "media.example.net",
        # Social handles
        "@stockphotos", "@imagepro", "@photohub", "@visualarts",
        "@creativemedia", "@picturepro", "@stockimages", "@photobank",
        # Copyright
        "© 2024", "© 2023", "© Sample Co", "© Image Ltd",
        "All Rights Reserved", "Do Not Copy", "For Review Only",
        # ID/codes
        "ID: 12345", "REF: ABC123", "CODE: XYZ789", "#STOCK2024",
        # Misc
        "NOT FOR RESALE", "EVALUATION COPY", "COMP IMAGE",
        "FOR POSITION ONLY", "FPO", "LAYOUT ONLY",
    ]

    # Validation texts (different style to test generalization)
    VAL_TEXTS = [
        # Different watermark words
        "Demo", "Test", "Temporary", "Pending", "Unregistered",
        "Trial", "Beta", "Mockup", "Placeholder", "Concept",
        # Different stock style
        "Great Photos", "Visual Plus", "Image World", "Photo Zone",
        "Video Vault", "Snap Gallery", "Frame Studio", "Lens Library",
        "Shot Archive", "Clip House", "Media Mart", "Picture Palace",
        # Different websites
        "www.testsite.com", "demo.photos.org", "trial.images.net",
        "beta.media.io", "sample.gallery.com", "preview.pics.org",
        # Different handles
        "@testphotos", "@demoimages", "@trialpics", "@betamedia",
        "@sampleshots", "@previewpix", "@mockupimg", "@placeholderpic",
        # Different copyright
        "© 2022", "© 2021", "© Demo Inc", "© Test Corp",
        "Rights Reserved", "Confidential", "Internal Use",
        # Different IDs
        "ID: 99999", "REF: TEST01", "CODE: DEMO42", "#TRIAL2024",
        # Different misc
        "NOT FINAL", "WORK IN PROGRESS", "DRAFT VERSION",
        "PREVIEW MODE", "SAMPLE ONLY", "TEMP FILE",
    ]

    # Combined for backward compatibility
    STOCK_TEXTS = TRAIN_TEXTS

    def __init__(
        self,
        frame_size: Tuple[int, int] = (256, 256),
        enabled_artifacts: Optional[Set[str]] = None,
        min_opacity: float = 0.5,
        max_opacity: float = 0.9,
        # Compression settings
        jpeg_quality_range: Tuple[int, int] = (10, 50),
        # Noise settings
        noise_std_range: Tuple[float, float] = (0.02, 0.15),
        # Mode for train/val text split
        mode: str = "train",
        # Size scale for logos/text (1.0 = default, 2.0 = double size)
        size_scale: float = 1.0,
    ):
        """
        Args:
            frame_size: (H, W) output frame size
            enabled_artifacts: Set of artifact types to use. If None, enables overlay types only.
            min_opacity: Minimum opacity for overlay artifacts
            max_opacity: Maximum opacity for overlay artifacts
            jpeg_quality_range: (min, max) JPEG quality for compression artifacts
            noise_std_range: (min, max) std for noise artifacts
            mode: "train" or "val" - uses different text sets for generalization testing
            size_scale: Scale factor for artifact sizes (1.0 = default 12-25% of width,
                        2.0 = double size, etc.). Affects logos, text, and patterns.
        """
        self.frame_size = frame_size  # (H, W)
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.jpeg_quality_range = jpeg_quality_range
        self.noise_std_range = noise_std_range
        self.mode = mode
        self.size_scale = size_scale

        # Text generator for diverse watermark text
        # Uses different seed for train vs val to ensure no overlap
        text_seed = 42 if mode == "train" else 12345
        self.text_generator = TextGenerator(seed=text_seed)
        self._text_call_count = 0  # Counter for deterministic generation

        # Legacy text lists (kept for backward compatibility)
        if mode == "val":
            self.texts = self.VAL_TEXTS
        else:
            self.texts = self.TRAIN_TEXTS

        # Default to overlay artifacts only (backward compatible)
        if enabled_artifacts is None:
            self.enabled_artifacts = self.OVERLAY_TYPES.copy()
        else:
            # Validate artifact types
            invalid = enabled_artifacts - self.ARTIFACT_TYPES
            if invalid:
                raise ValueError(f"Unknown artifact types: {invalid}")
            self.enabled_artifacts = enabled_artifacts

        # Separate enabled types
        self.enabled_overlays = self.enabled_artifacts & self.OVERLAY_TYPES
        self.enabled_degradations = self.enabled_artifacts & self.DEGRADATION_TYPES

        # Try to load fonts
        self.fonts = self._load_fonts()

    def _get_text(self) -> str:
        """Get diverse text using TextGenerator.

        Each call returns a different text deterministically based on call count.
        """
        text = self.text_generator.generate(self._text_call_count)
        self._text_call_count += 1
        return text

    def _load_fonts(self) -> Dict[str, List]:
        """Load available fonts at multiple sizes for per-example randomization.

        Returns dict with 'small', 'medium', 'large' keys, each containing a list of fonts.
        """
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
        ]

        # Base sizes scaled by size_scale
        sizes = {
            "small": int(16 * self.size_scale),
            "medium": int(24 * self.size_scale),
            "large": int(36 * self.size_scale),
        }

        fonts = {k: [] for k in sizes}

        for size_name, font_size in sizes.items():
            for path in font_paths:
                try:
                    fonts[size_name].append(ImageFont.truetype(path, font_size))
                except:
                    continue

            # Fallback to default if no fonts loaded
            if not fonts[size_name]:
                fonts[size_name].append(ImageFont.load_default())

        return fonts

    def _random_font(self):
        """Get a random font with random size."""
        size = random.choice(["small", "medium", "large"])
        return random.choice(self.fonts[size])

    def generate(
        self,
        num_frames: int,
        artifact_type: Optional[str] = None,
        clean_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Dict[str, Any]]:
        """
        Generate artifact for multiple frames.

        Args:
            num_frames: Number of frames
            artifact_type: Specific type to generate, or None for random from enabled
            clean_frames: [T, C, H, W] clean frames (required for degradation artifacts)

        Returns:
            artifact_mask: [T, 1, H, W] alpha mask (for overlay types) or None
            degraded_frames: [T, C, H, W] degraded frames (for degradation types) or None
            metadata: dict with artifact info
        """
        if artifact_type is None:
            artifact_type = random.choice(list(self.enabled_artifacts))

        if artifact_type not in self.enabled_artifacts:
            raise ValueError(f"Artifact type '{artifact_type}' is not enabled")

        # Route to appropriate generator
        if artifact_type in self.OVERLAY_TYPES:
            mask, metadata = self._generate_overlay(num_frames, artifact_type)
            return mask, None, metadata
        else:
            if clean_frames is None:
                raise ValueError(f"clean_frames required for artifact type '{artifact_type}'")
            degraded, metadata = self._generate_degradation(clean_frames, artifact_type)
            return None, degraded, metadata

    def generate_overlay(
        self,
        num_frames: int,
        artifact_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate overlay-type artifact (returns mask). Backward compatible API."""
        if artifact_type is None:
            if self.enabled_overlays:
                artifact_type = random.choice(list(self.enabled_overlays))
            else:
                raise ValueError("No overlay artifacts enabled")
        return self._generate_overlay(num_frames, artifact_type)

    def generate_degradation(
        self,
        clean_frames: torch.Tensor,
        artifact_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate degradation-type artifact (returns degraded frames)."""
        if artifact_type is None:
            if self.enabled_degradations:
                artifact_type = random.choice(list(self.enabled_degradations))
            else:
                raise ValueError("No degradation artifacts enabled")
        return self._generate_degradation(clean_frames, artifact_type)

    def _generate_overlay(
        self,
        num_frames: int,
        artifact_type: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate overlay-type artifacts (returns mask)"""
        generators = {
            "corner_logo": self._corner_logo,
            "text_overlay": self._text_overlay,
            "tiled_pattern": self._tiled_pattern,
            "moving_logo": self._moving_logo,
            "channel_logo": self._channel_logo,
            "diagonal_text": self._diagonal_text,
        }

        generator = generators[artifact_type]
        mask, metadata = generator(num_frames)
        metadata["type"] = artifact_type
        metadata["category"] = "overlay"

        return mask, metadata

    def _generate_degradation(
        self,
        clean_frames: torch.Tensor,
        artifact_type: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate degradation-type artifacts (returns degraded frames)"""
        generators = {
            "jpeg_compression": self._jpeg_compression,
            "video_compression": self._video_compression,
            "gaussian_noise": self._gaussian_noise,
            "salt_pepper_noise": self._salt_pepper_noise,
            "film_grain": self._film_grain,
            "color_banding": self._color_banding,
            "blur": self._blur,
        }

        generator = generators[artifact_type]
        degraded, metadata = generator(clean_frames)
        metadata["type"] = artifact_type
        metadata["category"] = "degradation"

        return degraded, metadata

    def _random_opacity(self) -> float:
        # Allow max_opacity > 1.0 to bias toward fully opaque
        # Values > 1.0 are clamped to 1.0
        opacity = random.uniform(self.min_opacity, self.max_opacity)
        return min(opacity, 1.0)

    # ==================== Overlay Artifacts ====================

    def _corner_logo(self, num_frames: int) -> Tuple[torch.Tensor, Dict]:
        """Static logo in corner"""
        H, W = self.frame_size
        opacity = self._random_opacity()

        # Base size: W//8 to W//4, scaled by size_scale
        base_min, base_max = W // 8, W // 4
        logo_size = int(random.randint(base_min, base_max) * self.size_scale)
        logo_size = min(logo_size, min(H, W) - 20)  # Don't exceed frame
        corner = random.choice(["tl", "tr", "bl", "br"])

        margin = random.randint(5, 20)
        if corner == "tl":
            x, y = margin, margin
        elif corner == "tr":
            x, y = W - logo_size - margin, margin
        elif corner == "bl":
            x, y = margin, H - logo_size - margin
        else:  # br
            x, y = W - logo_size - margin, H - logo_size - margin

        img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(img)

        shape = random.choice(["rectangle", "ellipse", "text"])
        if shape == "rectangle":
            draw.rectangle([x, y, x + logo_size, y + logo_size], fill=int(255 * opacity))
        elif shape == "ellipse":
            draw.ellipse([x, y, x + logo_size, y + logo_size], fill=int(255 * opacity))
        else:
            text = self._get_text()
            font = self._random_font()
            draw.text((x, y), text, fill=int(255 * opacity), font=font)

        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        mask = torch.from_numpy(np.array(img)).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(num_frames, 1, 1, 1)

        return mask, {"corner": corner, "opacity": opacity}

    def _text_overlay(self, num_frames: int) -> Tuple[torch.Tensor, Dict]:
        """Text overlay"""
        H, W = self.frame_size
        opacity = self._random_opacity()

        img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(img)

        text = self._get_text()
        font = self._random_font()

        x = random.randint(0, max(1, W // 2))
        y = random.randint(0, max(1, H // 2))

        draw.text((x, y), text, fill=int(255 * opacity), font=font)

        mask = torch.from_numpy(np.array(img)).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_frames, 1, 1, 1)

        return mask, {"text": text, "opacity": opacity}

    def _tiled_pattern(self, num_frames: int) -> Tuple[torch.Tensor, Dict]:
        """Repeating tiled pattern"""
        H, W = self.frame_size
        opacity = self._random_opacity()

        img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(img)

        text = self._get_text()
        font = self._random_font()

        spacing_x = random.randint(80, 150)
        spacing_y = random.randint(60, 100)
        rotation = random.choice([0, -30, -45, 30, 45])

        for y_pos in range(-H, H * 2, spacing_y):
            for x_pos in range(-W, W * 2, spacing_x):
                draw.text((x_pos, y_pos), text, fill=int(255 * opacity), font=font)

        if rotation != 0:
            img = img.rotate(rotation, resample=Image.BILINEAR, expand=False)
            img = img.crop(((img.width - W) // 2, (img.height - H) // 2,
                           (img.width + W) // 2, (img.height + H) // 2))

        mask = torch.from_numpy(np.array(img)).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_frames, 1, 1, 1)

        return mask, {"text": text, "rotation": rotation, "opacity": opacity}

    def _moving_logo(self, num_frames: int) -> Tuple[torch.Tensor, Dict]:
        """Logo that moves across frames"""
        H, W = self.frame_size
        opacity = self._random_opacity()

        # Base size: W//8 to W//5, scaled by size_scale
        base_min, base_max = W // 8, W // 5
        logo_size = int(random.randint(base_min, base_max) * self.size_scale)
        logo_size = min(logo_size, min(H, W) // 2)  # Don't exceed half frame
        pattern = random.choice(["linear", "bounce", "circular"])

        start_x = random.randint(0, W - logo_size)
        start_y = random.randint(0, H - logo_size)
        speed_x = random.uniform(-5, 5)
        speed_y = random.uniform(-3, 3)

        masks = []
        for t in range(num_frames):
            img = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(img)

            if pattern == "linear":
                x = int(start_x + speed_x * t) % (W - logo_size)
                y = int(start_y + speed_y * t) % (H - logo_size)
            elif pattern == "bounce":
                x = int(start_x + speed_x * t)
                y = int(start_y + speed_y * t)
                x = abs(x % (2 * (W - logo_size)) - (W - logo_size))
                y = abs(y % (2 * (H - logo_size)) - (H - logo_size))
            else:  # circular
                angle = 2 * math.pi * t / max(num_frames, 1)
                radius = min(W, H) // 6
                x = int(W // 2 + radius * math.cos(angle) - logo_size // 2)
                y = int(H // 2 + radius * math.sin(angle) - logo_size // 2)

            x = max(0, min(x, W - logo_size))
            y = max(0, min(y, H - logo_size))

            draw.ellipse([x, y, x + logo_size, y + logo_size], fill=int(255 * opacity))

            mask = torch.from_numpy(np.array(img)).float() / 255.0
            masks.append(mask)

        masks = torch.stack(masks).unsqueeze(1)

        return masks, {"pattern": pattern, "opacity": opacity}

    def _channel_logo(self, num_frames: int) -> Tuple[torch.Tensor, Dict]:
        """TV channel style logo"""
        H, W = self.frame_size
        opacity = self._random_opacity()

        img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(img)

        text = random.choice(["CH", "TV", "HD", "4K", "LIVE", "REC"])
        font = self._random_font()

        corner = random.choice(["tr", "br"])
        margin = 10

        if corner == "tr":
            x, y = W - 50 - margin, margin
        else:
            x, y = W - 50 - margin, H - 30 - margin

        draw.text((x, y), text, fill=int(255 * opacity), font=font)

        mask = torch.from_numpy(np.array(img)).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_frames, 1, 1, 1)

        return mask, {"text": text, "corner": corner, "opacity": opacity}

    def _diagonal_text(self, num_frames: int) -> Tuple[torch.Tensor, Dict]:
        """Large diagonal text across frame"""
        H, W = self.frame_size
        opacity = self._random_opacity()

        diag = int(math.sqrt(H**2 + W**2))
        img = Image.new('L', (diag, diag), 0)
        draw = ImageDraw.Draw(img)

        text = self._get_text()
        font = self._random_font()

        draw.text((diag // 4, diag // 2), text, fill=int(255 * opacity), font=font)

        angle = random.choice([-45, -30, 30, 45])
        img = img.rotate(angle, resample=Image.BILINEAR)

        left = (diag - W) // 2
        top = (diag - H) // 2
        img = img.crop((left, top, left + W, top + H))

        mask = torch.from_numpy(np.array(img)).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_frames, 1, 1, 1)

        return mask, {"text": text, "angle": angle, "opacity": opacity}

    # ==================== Degradation Artifacts ====================

    def _jpeg_compression(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate JPEG compression artifacts"""
        import io
        quality = random.randint(*self.jpeg_quality_range)

        degraded = []
        for frame in frames:
            # Convert to PIL
            img = Image.fromarray(
                (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

            # Compress via JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer)

            # Convert back
            arr = np.array(compressed).astype(np.float32) / 255.0
            degraded.append(torch.from_numpy(arr).permute(2, 0, 1))

        return torch.stack(degraded), {"quality": quality}

    def _video_compression(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate video codec artifacts (blocking, banding)"""
        quality = random.randint(15, 40)
        block_size = random.choice([8, 16])

        degraded = []
        for frame in frames:
            C, H, W = frame.shape
            blocked = frame.clone()

            for i in range(0, H, block_size):
                for j in range(0, W, block_size):
                    block = blocked[:, i:i+block_size, j:j+block_size]
                    levels = random.randint(16, 64)
                    block_q = (block * levels).round() / levels
                    blocked[:, i:i+block_size, j:j+block_size] = block_q

            degraded.append(blocked)

        return torch.stack(degraded), {"quality": quality, "block_size": block_size}

    def _gaussian_noise(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Add Gaussian noise"""
        std = random.uniform(*self.noise_std_range)

        noise = torch.randn_like(frames) * std
        degraded = (frames + noise).clamp(0, 1)

        return degraded, {"std": std}

    def _salt_pepper_noise(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Add salt and pepper noise"""
        density = random.uniform(0.01, 0.05)

        degraded = frames.clone()
        mask = torch.rand_like(frames[:, :1, :, :])

        salt_mask = mask < density / 2
        degraded = torch.where(salt_mask.expand_as(degraded), torch.ones_like(degraded), degraded)

        pepper_mask = mask > 1 - density / 2
        degraded = torch.where(pepper_mask.expand_as(degraded), torch.zeros_like(degraded), degraded)

        return degraded, {"density": density}

    def _film_grain(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Add film grain effect"""
        intensity = random.uniform(0.02, 0.1)

        grain = torch.randn_like(frames) * intensity

        luminance = 0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
        grain_strength = 4 * luminance * (1 - luminance)

        degraded = (frames + grain * grain_strength).clamp(0, 1)

        return degraded, {"intensity": intensity}

    def _color_banding(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate color banding (reduced bit depth)"""
        levels = random.choice([8, 16, 32])

        degraded = (frames * levels).round() / levels

        return degraded, {"levels": levels}

    def _blur(self, frames: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Add blur artifact"""
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)

        x = torch.arange(kernel_size) - kernel_size // 2
        kernel_1d = torch.exp(-x.float()**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

        T, C, H, W = frames.shape
        degraded = frames.view(T * C, 1, H, W)
        padding = kernel_size // 2
        degraded = F.conv2d(degraded, kernel_2d.to(frames.device), padding=padding)
        degraded = degraded.view(T, C, H, W)

        return degraded, {"kernel_size": kernel_size, "sigma": sigma}


def apply_overlay_artifact(
    frames: torch.Tensor,
    artifact_mask: torch.Tensor,
    color: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply overlay artifact to frames.

    Args:
        frames: [T, C, H, W] or [B, T, C, H, W] clean frames
        artifact_mask: [T, 1, H, W] artifact alpha mask
        color: [3] artifact color, default white/gray

    Returns:
        degraded frames same shape as input
    """
    if color is None:
        if random.random() < 0.7:
            color = torch.ones(3)
        else:
            gray = random.uniform(0.5, 1.0)
            color = torch.full((3,), gray)

    has_batch = frames.dim() == 5
    if has_batch:
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        artifact_mask = artifact_mask.unsqueeze(0).expand(B, -1, -1, -1, -1)
        artifact_mask = artifact_mask.reshape(B * T, 1, H, W)

    color = color.to(frames.device)
    artifact_mask = artifact_mask.to(frames.device)

    # Alpha compositing: result = background * (1 - alpha) + foreground * alpha
    overlay = color.view(1, 3, 1, 1)
    result = frames * (1 - artifact_mask) + overlay * artifact_mask

    if has_batch:
        result = result.view(B, T, C, H, W)

    return result




# Test
if __name__ == "__main__":
    print("Testing TextGenerator...")
    text_gen = TextGenerator(seed=42)
    texts = text_gen.generate_batch(0, 20)
    print(f"  Generated {len(texts)} unique texts:")
    for i, t in enumerate(texts[:10]):
        print(f"    {i}: {t}")
    # Verify determinism
    texts2 = text_gen.generate_batch(0, 10)
    assert texts[:10] == texts2, "TextGenerator should be deterministic"
    print("  Determinism verified!")

    print("\nTesting ArtifactGenerator...")

    # Test overlay artifacts (default)
    gen = ArtifactGenerator(frame_size=(256, 256))
    print("\nOverlay artifacts:")
    for atype in ["corner_logo", "text_overlay", "tiled_pattern", "moving_logo"]:
        mask, degraded, meta = gen.generate(5, artifact_type=atype)
        print(f"  {atype}: mask shape {mask.shape}, meta: {meta}")

    # Test degradation artifacts
    gen_full = ArtifactGenerator(
        frame_size=(256, 256),
        enabled_artifacts={
            "gaussian_noise", "jpeg_compression", "blur",
            "color_banding", "film_grain", "corner_logo"
        }
    )

    clean = torch.rand(5, 3, 256, 256)
    print("\nDegradation artifacts:")
    for atype in ["gaussian_noise", "blur", "color_banding", "film_grain"]:
        mask, degraded, meta = gen_full.generate(5, artifact_type=atype, clean_frames=clean)
        print(f"  {atype}: degraded shape {degraded.shape}, meta: {meta}")

    # Test apply_overlay_artifact
    print("\nTesting apply_overlay_artifact...")
    mask, _, _ = gen.generate(5, artifact_type="corner_logo")
    result = apply_overlay_artifact(clean, mask)
    print(f"  Input: {clean.shape}, Output: {result.shape}")

    print("\nAll tests passed!")
