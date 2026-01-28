---
title: DocVision IQ
emoji: 🏆
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---


# 📄 DocVision IQ

**Employee Name:** `<EmpID>`

---

## 1. Research Question / Hypothesis

### Research Question
Can a hybrid pipeline combining OCR, Vision Large Language Models (Vision LLMs), and visual cue detection accurately classify and understand real-world document images and PDFs in a production-style API system?

### Hypothesis
Integrating high-quality OCR (LlamaParse), Vision LLM reasoning, and logo/seal detection will improve document classification robustness compared to OCR-only or vision-only approaches, especially for visually distinctive documents.

---

## 2. Motivation and Relevance

Organizations handle large volumes of unstructured documents such as invoices, identity cards, certificates, and contracts. Traditional OCR-only systems struggle with:

- Diverse document layouts  
- Poor scan quality  
- Visual identifiers (logos, seals, emblems)  
- Contextual ambiguity  

**DocVision** addresses these challenges using a **multi-modal document understanding pipeline**, closely reflecting real-world enterprise document intelligence systems used in **fintech, compliance, onboarding, and automation workflows**.

---

## 3. System Architecture

DocVision is implemented as a **modular, production-style system** with clear separation of concerns.

### High-Level Architecture

```

User (UI / API)
↓
FastAPI Backend
↓
Validation & Hashing
↓
PDF → Image Conversion (PyMuPDF)
↓
OCR (LlamaParse)
↓
Vision LLM Classification (OpenRouter)
↓
Visual Cue Detection (Logos / Seals)
↓
Caching Layer
↓
Structured JSON Output
```

**Backend:** FastAPI (`src/main.py`)  
**Frontend:** Streamlit (`frontend/app.py`)  
**Deployment:** Docker  
**Experiments:** Declarative YAML (`experiments/`)

---

## 4. Models and Versions Used

### OCR
- **LlamaParse (Llama Cloud Services)**
  - Used for high-quality text extraction from images and PDFs

### Vision LLM
- **nvidia/nemotron-nano-12b-v2-vl** (via OpenRouter)
  - Used for document classification and reasoning using combined text and image inputs

### Visual Cue Detection
- **ellabettison/Logo-Detection-finetune**
  - Transformer-based object detection model for detecting logos and seals

---

## 5. Prompting and / or Fine-Tuning Strategy

- **Zero-shot prompting** (no fine-tuning)
- Carefully designed **instruction-based prompt** that:
  - Enforces strict JSON output
  - Prioritizes strong document-specific identifiers
  - Includes explicit classification constraints (e.g., Aadhaar rules)
  - Combines OCR text with image context

**Rationale:**  
Zero-shot prompting ensures better generalization and aligns with real-world Vision LLM API usage without introducing dataset-specific bias.

---

## 6. Evaluation Protocol

Evaluation is performed using a combination of **automated** and **human** methods.

### Automated Evaluation
- JSON schema validation
- Rule-based checks (e.g., Aadhaar number presence)
- Field extraction completeness
- End-to-end latency measurement

### Human Evaluation
- Manual inspection of document type correctness
- Assessment of reasoning quality and plausibility
- Evaluation of visual cue relevance

Experiments are defined declaratively in YAML files  
(`experiments/exp_01.yaml`, `experiments/exp_02.yaml`) to ensure reproducibility.

---

## 7. Key Results

- Consistent document classification across common document categories
- Improved robustness when visual cue detection is enabled
- Stable performance on scanned images and PDFs
- Deterministic preprocessing and bounded runtime

(Refer to the **Experiments** and **Reproducibility Statement** for detailed analysis.)

---

## 8. Known Limitations and Ethical Considerations

### Limitations
- Performance degrades on extremely low-resolution or heavily occluded documents
- Dependence on external APIs for OCR and Vision LLM inference
- Field-level extraction accuracy is not benchmarked against labeled datasets

### Ethical Considerations
- Handles potentially sensitive personal documents
- No data is permanently stored beyond processing
- API keys are required and must be securely managed
- System outputs should not be used for identity verification without human review

---

## 9. Exact Instructions to Reproduce Results

### 9.1 Prerequisites
- Python 3.10+
- Docker installed
- Internet access (required for external APIs)

### 9.2 Environment Configuration

Create a `.env` file in the project root to securely store API credentials:

```env
LLAMA_API_KEY=llx-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
```

### 9.3 Project Structure

```text
DOCVISION_IQ/
│
├── experiments/
│   ├── exp_01.yaml
│   └── exp_02.yaml
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── config.py
│   ├── main.py
│   ├── pdfconverter.py
│   ├── textextraction.py
│   ├── vision.py
│   └── visual_cues.py
│
├── uploads/
│
├── Dockerfile
├── project.yaml
├── reproducibility.md
├── requirements.txt
└── README.md
```

### 9.4 Docker Execution

Build the Docker image:

```bash
docker build -t docvision .
docker run -p 8000:8000 -p 8501:8501 --env-file .env docvision
```

### 9.5 Access

| Component | URL |
|----------|-----|
| **Streamlit UI** | http://localhost:8501 |
| **FastAPI Docs** | http://localhost:8000/docs |

---

## 👨‍💻 Author

**DocVision**  
An end-to-end AI-powered document understanding system built for  
**real-world applications, interviews, and scalable deployments**.

