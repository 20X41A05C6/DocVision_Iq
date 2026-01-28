

import os
import base64
import json
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from src.textextraction import extract_text_from_image_async
from src.config import VISION_MODEL_NAME

# --------------------------------------------------
# ENVIRONMENT SETUP
# --------------------------------------------------
# Load environment variables for API keys
load_dotenv()


# --------------------------------------------------
# OPENROUTER CLIENT INITIALIZATION
# --------------------------------------------------
client: OpenAI = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# STRICT JSON PROMPT (UNCHANGED)
# --------------------------------------------------
PROMPT = """
You are an intelligent document understanding system.
**Prompt:** You are an advanced document classification AI tasked with accurately identifying the specific type of document presented to Your objective is to analyze the visual layout and textual content of the document while adhering to the following guidelines: 
1.**Visual Layout Analysis**: Examine the structural elements of the document such as logos, headers, footers, and any unique formatting that may indicate the document type.Pay attention to layout patterns that are characteristic of each document type.
2.**Textual Evidence Extraction**: Extract and analyze textual information from the document.Look for key phrases, terms, and identifiers that are strongly associated with each document type.This includes: - For passports: Look for terms like "Passport", "Nationality", "Date of Birth", and country-specific formatting.- For aadhaar cards: Identify "Aadhaar Number", "Biometric Data", and any unique UIDAI branding.- For pan cards: Search for "Permanent Account Number" and tax-related keywords.- For contracts: Identify terms like "Agreement", "Parties", "Terms and Conditions".- For invoices: Look for "Invoice Number", "Billing Address", "Total Amount".
3.**Prioritization of Strong Identifiers**: Focus on strong, document-specific identifiers rather than generic text.If there is ambiguity or multiple potential matches, prioritize the identifiers that are most definitive and unique to the document type.
4.**Avoiding Guesses**: Do not make assumptions or guesses if the evidence is insufficient.If the document does not clearly fit into one of the specified categories based on the analysis, return a response indicating that the document type cannot be determined with confidence.
5.**Output Format**: Provide your classification result in the following format: - "Document Type: [identified type]".

important: Your analysis must be thorough and based on concrete evidence from the document's content and layout.
there is a difference between invoice and receipt.etc 

extract exach and every information from the document.
and provide correct field names and values

important:Never classify a document as aadhaar_card unless a clear 12-digit Aadhaar number OR UIDAI reference is present.
extract fields each and every important information.

this is normal reasoning brief 2 to 3 lines explanation of how the document type was determined,Highlight underline

take document type decision based on the normal reasoning.

OUTPUT FORMAT (JSON – STRICT, ADDITIONAL):

Return VALID JSON ONLY.
Do not include any text outside JSON.
Do not include markdown or code blocks.

{
  "document_type": "<Document Type>",
  "reasoning": "<brief 2 to 3 lines explanation of how the document type was determined, highlighting the key visual or textual features that influenced the decision>",
  "extracted_textfields": {
    "<field_name>": "<value>",
    "<field_name>": "<value>"
  }
}
"""


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def image_to_base64(path: str) -> str:
    """
    Convert an image file to a base64-encoded string.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    str
        Base64-encoded image content.
    """
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Safely extract JSON content from model output.

    This function handles cases where the model may
    accidentally include extra text around the JSON.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If valid JSON cannot be extracted.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise ValueError("LLM did not return valid JSON")


# --------------------------------------------------
# ASYNC DOCUMENT CLASSIFICATION
# --------------------------------------------------
async def classify_image(image_path: str) -> Dict[str, Any]:
    """
    Perform document classification using OCR + Vision LLM.

    Steps:
    1. Extract text using OCR
    2. Encode image as base64
    3. Send text + image to Vision LLM
    4. Parse and normalize JSON output

    Parameters
    ----------
    image_path : str
        Path to the image or PDF page.

    Returns
    -------
    dict
        Structured classification result with document type,
        reasoning, and extracted fields.
    """
    # OCR extraction
    ocr_text: str = await extract_text_from_image_async(image_path)

    # Image encoding
    image_base64: str = image_to_base64(image_path)

    # Vision LLM request
    response = client.chat.completions.create(
        model=VISION_MODEL_NAME,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT + "\n\nOCR TEXT:\n" + ocr_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    )

    raw_output: str = response.choices[0].message.content.strip()

    # Safe JSON parsing
    try:
        result: Dict[str, Any] = extract_json_from_text(raw_output)
    except Exception:
        result = {
            "document_type": "unknown",
            "reasoning": "Model output could not be parsed as JSON",
            "extracted_textfields": {}
        }

    # Ensure required keys are always present
    return {
        "document_type": result.get("document_type", "unknown"),
        "reasoning": result.get("reasoning", ""),
        "extracted_textfields": result.get("extracted_textfields", {}),
    }
