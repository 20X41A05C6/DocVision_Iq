
import os
import io
import hashlib
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.pdfconverter import pdf_to_images
from src.vision import classify_image
from src.visual_cues import detect_logos_from_bytes
from src.config import (
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
    MAX_TOTAL_FILES,
    MAX_PDFS,
    MAX_IMAGES,
    MAX_IMAGE_MB,
    MAX_PDF_MB,
    MIN_WIDTH,
    MIN_HEIGHT,
    MAX_WIDTH,
    MAX_HEIGHT,
    MAX_VISUAL_PAGES,
    MAX_LOGOS_PER_PAGE,
    MAX_IMAGE_RESIZE,
)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------
app = FastAPI(title="DocVision API")


# --------------------------------------------------
# CORS (REQUIRED FOR HUGGING FACE)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health() -> Dict[str, str]:
    """Health check endpoint for routing and monitoring."""
    return {"status": "ok"}


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# IN-MEMORY CACHES
# --------------------------------------------------
TEXT_CACHE: Dict[str, Dict[str, Any]] = {}
VISUAL_CACHE: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def file_hash(data: bytes) -> str:
    """Generate a deterministic hash for file contents."""
    return hashlib.md5(data).hexdigest()


def read_file(file: UploadFile) -> bytes:
    """Read file contents without consuming the stream."""
    data = file.file.read()
    file.file.seek(0)
    return data


def validate_file(file: UploadFile, contents: bytes) -> str | None:
    """
    Validate file type, size, and image resolution.

    Returns an error message if invalid, otherwise None.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    size_mb = len(contents) / (1024 * 1024)

    if ext not in ALLOWED_EXTENSIONS:
        return "Unsupported file format"

    if ext == ".pdf" and size_mb > MAX_PDF_MB:
        return f"PDF exceeds {MAX_PDF_MB} MB"

    if ext != ".pdf" and size_mb > MAX_IMAGE_MB:
        return f"Image exceeds {MAX_IMAGE_MB} MB"

    if ext != ".pdf":
        try:
            image = Image.open(io.BytesIO(contents))
            width, height = image.size

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return f"Image too small ({width}x{height})"

            if width > MAX_WIDTH or height > MAX_HEIGHT:
                return f"Image too large ({width}x{height})"

        except Exception:
            return "Invalid image file"

    return None


# --------------------------------------------------
# DOCUMENT ANALYSIS ENDPOINT
# --------------------------------------------------
@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Perform OCR + Vision-based document classification.
    """
    if len(files) > MAX_TOTAL_FILES:
        return JSONResponse(
            {"error": f"Maximum {MAX_TOTAL_FILES} files allowed"},
            status_code=400,
        )

    pdf_count = sum(f.filename.lower().endswith(".pdf") for f in files)
    img_count = len(files) - pdf_count

    async def process_file(file: UploadFile) -> Dict[str, Any]:
        contents = read_file(file)
        fid = f"{file.filename}_{file_hash(contents)}"

        if file.filename.lower().endswith(".pdf") and pdf_count > MAX_PDFS:
            return {"file": file.filename, "error": f"Maximum {MAX_PDFS} PDFs allowed"}

        if not file.filename.lower().endswith(".pdf") and img_count > MAX_IMAGES:
            return {"file": file.filename, "error": f"Maximum {MAX_IMAGES} images allowed"}

        if fid in TEXT_CACHE:
            return TEXT_CACHE[fid]

        error = validate_file(file, contents)
        if error:
            return {"file": file.filename, "error": error}

        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(contents)

        try:
            if file.filename.lower().endswith(".pdf"):
                pdf_name = await asyncio.to_thread(pdf_to_images, path)
                base_dir = os.path.join("uploads", "images", pdf_name)
                first_page = sorted(os.listdir(base_dir))[0]
                analysis = await classify_image(os.path.join(base_dir, first_page))
            else:
                analysis = await classify_image(path)

            result = {
                "file": file.filename,
                "document_type": analysis.get("document_type"),
                "reasoning": analysis.get("reasoning"),
                "extracted_textfields": analysis.get("extracted_textfields", {}),
            }

            TEXT_CACHE[fid] = result
            return result

        except Exception as exc:
            return {"file": file.filename, "error": f"Processing failed: {exc}"}

    results = await asyncio.gather(*[process_file(f) for f in files])
    return JSONResponse(content=results)


# --------------------------------------------------
# VISUAL CUES ENDPOINT
# --------------------------------------------------
@app.post("/visual_cues")
async def visual_cues(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Detect logos, seals, and visual symbols from documents.
    """

    async def process_visual(file: UploadFile) -> Dict[str, Any]:
        contents = read_file(file)
        fid = f"{file.filename}_{file_hash(contents)}"

        if fid in VISUAL_CACHE:
            return VISUAL_CACHE[fid]

        error = validate_file(file, contents)
        if error:
            return {"file": file.filename, "error": error}

        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(contents)

        visuals = []

        try:
            if file.filename.lower().endswith(".pdf"):
                pdf_name = await asyncio.to_thread(pdf_to_images, path)
                base_dir = os.path.join("uploads", "images", pdf_name)

                for img_name in sorted(os.listdir(base_dir))[:MAX_VISUAL_PAGES]:
                    with open(os.path.join(base_dir, img_name), "rb") as img_file:
                        logos = await asyncio.to_thread(
                            detect_logos_from_bytes,
                            img_file.read(),
                            MAX_IMAGE_RESIZE,
                            MAX_LOGOS_PER_PAGE,
                        )
                    visuals.append({"page": img_name, "logos": logos})
            else:
                logos = await asyncio.to_thread(
                    detect_logos_from_bytes,
                    contents,
                    MAX_IMAGE_RESIZE,
                    MAX_LOGOS_PER_PAGE,
                )
                visuals.append({"page": "image", "logos": logos})

            result = {"file": file.filename, "visual_cues": visuals}
            VISUAL_CACHE[fid] = result
            return result

        except Exception as exc:
            return {"file": file.filename, "error": f"Visual processing failed: {exc}"}

    results = await asyncio.gather(*[process_visual(f) for f in files])
    return JSONResponse(content=results)
