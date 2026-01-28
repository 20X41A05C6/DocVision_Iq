
import io
import base64
from typing import List, Dict, Tuple

from PIL import Image
from transformers import pipeline

from src.config import LOGO_DETECTION_MODEL


# --------------------------------------------------
# MODEL INITIALIZATION (LOAD ONCE)
# --------------------------------------------------
# Object detection pipeline for logo / seal detection
detector = pipeline(
    task="object-detection",
    model=LOGO_DETECTION_MODEL,
    device=-1  # CPU
)


# --------------------------------------------------
# LOGO DETECTION
# --------------------------------------------------
def detect_logos_from_bytes(
    image_bytes: bytes,
    resize: Tuple[int, int] = (1024, 1024),
    max_logos: int = 3
) -> List[Dict[str, str | float]]:
    """
    Detect logos or visual emblems from raw image bytes.

    The function resizes the image for faster inference,
    detects logo regions, crops them, and returns the
    cropped logo images encoded in base64 along with
    confidence scores.

    Parameters
    ----------
    image_bytes : bytes
        Raw image data.
    resize : tuple[int, int], optional
        Maximum image size for inference (default: 1024x1024).
    max_logos : int, optional
        Maximum number of detected logos to return.

    Returns
    -------
    list[dict]
        List of detected logos with:
        - confidence: float
        - image_base64: str
    """

    # Load image from bytes
    image: Image.Image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize image for performance optimization
    image.thumbnail(resize)

    # Run object detection
    detections = detector(image)

    results: List[Dict[str, str | float]] = []

    # Process top detections only
    for det in detections[:max_logos]:
        box = det["box"]
        score: float = float(det["score"])

        xmin: int = int(box["xmin"])
        ymin: int = int(box["ymin"])
        xmax: int = int(box["xmax"])
        ymax: int = int(box["ymax"])

        # Crop detected logo region
        cropped = image.crop((xmin, ymin, xmax, ymax))

        # Convert cropped logo to base64
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")

        results.append({
            "confidence": round(score, 3),
            "image_base64": base64.b64encode(buffer.getvalue()).decode()
        })

    return results
