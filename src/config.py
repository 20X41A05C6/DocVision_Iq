# Maximum number of concurrent OCR requests
MAX_CONCURRENT_OCR: int = 5

# Vision LLM model name (OpenRouter)
VISION_MODEL_NAME: str = "nvidia/nemotron-nano-12b-v2-vl:free"


# Logo detection model
LOGO_DETECTION_MODEL: str = "ellabettison/Logo-Detection-finetune"


# PDF to image conversion settings
PDF_IMAGE_DPI: int = 302
PDF_IMAGE_BASE_DIR: str = "uploads/images"


# file upload limitations

UPLOAD_DIR: str = "uploads"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

MAX_TOTAL_FILES: int = 5
MAX_PDFS: int = 3
MAX_IMAGES: int = 5

MAX_IMAGE_MB: int = 5
MAX_PDF_MB: int = 10

# -------------------------------
# IMAGE VALIDATION
# -------------------------------
MIN_WIDTH: int = 300
MIN_HEIGHT: int = 300
MAX_WIDTH: int = 6000
MAX_HEIGHT: int = 6000

# -------------------------------
# VISUAL CUES
# -------------------------------
MAX_VISUAL_PAGES: int = 1
MAX_LOGOS_PER_PAGE: int = 4
MAX_IMAGE_RESIZE = (1024, 1024)

