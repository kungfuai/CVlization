"""Layout detector for OMR pages.

Multi-class detection of {system, staff, barline, [key_signature, clef]}
on a rendered score page. Cells (staff × measure) are derived from the
combination of staff bboxes and barline x-positions — not detected
directly.

Choice of backbone is open:
  - Ultralytics YOLOv8/YOLOv11 (small models, mature tooling, easiest start).
  - DETR/RT-DETR (no NMS, attention-based).
  - Custom minimal head on a ResNet backbone (most control).

For first iteration: small Ultralytics-style YOLO is the path of least
resistance. The detection problem on rendered sheet music is *very easy*
visually (high contrast, regular structures), so a tiny model should suffice.

Status: STUB — interface only. Pick a backbone after the label-extraction
step is unblocked.

Public API (planned):
    model = LayoutDetector(num_classes=5).train(dataset, epochs, ...)
    predictions = model.predict(image) -> dict like extract_bboxes' output

Expected confusion matrix: extremely high diagonal for staves/systems
(deterministic visual structure), good for barlines, slightly more
ambiguity on key-sig glyph bboxes due to clustering.
"""

CLASSES = ("system", "staff", "barline", "key_signature", "clef")


class LayoutDetector:
    """Stub. Implement with a chosen backbone."""

    def __init__(self, num_classes: int = len(CLASSES), backbone: str = "yolov8n"):
        self.num_classes = num_classes
        self.backbone = backbone
        raise NotImplementedError("Choose backbone + implement.")

    def predict(self, image) -> dict:
        """image (PIL or numpy) -> dict of detected layout (same shape as
        labels/extract_bboxes.extract_layout output)."""
        raise NotImplementedError
