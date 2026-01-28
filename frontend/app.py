import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

# --------------------------------------------------
# INTERNAL API (HF SAFE)
# --------------------------------------------------
API_URL = "http://localhost:8000"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="DocVision IQ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LIMITS (MUST MATCH BACKEND)
# --------------------------------------------------
MAX_TOTAL_FILES = 5
MAX_IMAGES = 5
MAX_PDFS = 3

# --------------------------------------------------
# STYLES
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #1a1a1a;
}
.file-header {
    background-color: #000000;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 12px;
}
.section-gap {
    margin-bottom: 18px;
}
.logo-card {
    background-color: #fafafa;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #cccccc;
    text-align: center;
}
.confidence {
    color: #0b6623;
    font-size: 12px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📄 DocVision IQ")
st.caption("AI-powered Document Understanding & Visual Cue Extraction")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Images or PDFs",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

extract_visual = st.checkbox("🧿 Extract Visual Cues (Logos / Seals)")

# --------------------------------------------------
# CLIENT-SIDE VALIDATION
# --------------------------------------------------
if uploaded_files:
    total = len(uploaded_files)
    pdfs = sum(1 for f in uploaded_files if f.name.lower().endswith(".pdf"))
    images = total - pdfs

    if total > MAX_TOTAL_FILES:
        st.error(f"❌ Maximum {MAX_TOTAL_FILES} files allowed")
        st.stop()

    if pdfs > MAX_PDFS:
        st.error(f"❌ Maximum {MAX_PDFS} PDFs allowed")
        st.stop()

    if images > MAX_IMAGES:
        st.error(f"❌ Maximum {MAX_IMAGES} images allowed")
        st.stop()

    files = [
        ("files", (file.name, file.getvalue(), file.type))
        for file in uploaded_files
    ]

    # --------------------------------------------------
    # ANALYZE DOCUMENTS
    # --------------------------------------------------
    with st.spinner("🔍 Analyzing documents..."):
        response = requests.post(
            f"{API_URL}/analyze",
            files=files,
            timeout=300
        )

    if response.status_code != 200:
        try:
            st.error(response.json().get("message", "Analyze API failed"))
        except Exception:
            st.error("Analyze API failed")
        st.stop()

    analysis_data = response.json()

    # --------------------------------------------------
    # VISUAL CUES
    # --------------------------------------------------
    visual_map = {}

    if extract_visual:
        with st.spinner("🧿 Extracting visual cues..."):
            visual_response = requests.post(
                f"{API_URL}/visual_cues",
                files=files,
                timeout=300
            )

        if visual_response.status_code != 200:
            try:
                st.error(visual_response.json().get("message", "Visual cues API failed"))
            except Exception:
                st.error("Visual cues API failed")
            st.stop()

        visual_data = visual_response.json()
        for item in visual_data:
            visual_map[item["file"]] = item.get("visual_cues", [])

    # --------------------------------------------------
    # RENDER RESULTS
    # --------------------------------------------------
    for item in analysis_data:
        filename = item.get("file", "Unknown File")

        st.markdown(
            f"<div class='file-header'>📄 {filename}</div>",
            unsafe_allow_html=True
        )

        if "error" in item:
            st.error(item["error"])
            continue

        st.markdown(
            f"<div class='section-gap'><strong>📌 Document Type:</strong> {item.get('document_type')}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='section-gap'><strong>Reasoning:</strong><br>{item.get('reasoning')}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='section-gap'><strong>Extracted Text Fields:</strong></div>",
            unsafe_allow_html=True
        )

        fields = item.get("extracted_textfields", {})
        if not fields:
            st.info("No text fields extracted")
        else:
            for k, v in fields.items():
                st.markdown(f"- **{k}**: {v}")

        if extract_visual and filename in visual_map:
            logos = []
            for page in visual_map[filename]:
                logos.extend(page.get("logos", []))

            if logos:
                cols = st.columns(min(len(logos), 4))
                for col, logo in zip(cols, logos):
                    with col:
                        img = Image.open(
                            io.BytesIO(base64.b64decode(logo["image_base64"]))
                        )
                        st.markdown("<div class='logo-card'>", unsafe_allow_html=True)
                        st.image(img)
                        st.markdown(
                            f"<div class='confidence'>Confidence: {logo['confidence']}</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No visual cues found")
