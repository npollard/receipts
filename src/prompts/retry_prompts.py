"""Reusable prompt templates for receipt parsing retry strategies"""

LLM_FIX_PROMPT = """
You are correcting a receipt parsing error.

The previous structured output failed validation:

ERROR:
{validation_error}

OCR TEXT:
{ocr_text}

PREVIOUS OUTPUT:
{previous_output}

Instructions:
- Fix output so that TOTAL matches sum of items
- Ensure all items are included
- Do NOT invent items not present in OCR text
- Prefer correcting item extraction over changing total unless clearly wrong

Return ONLY valid JSON in the same schema:
{{
    "date": "YYYY-MM-DD",
    "total": 123.45,
    "items": [
        {{
            "description": "Item description",
            "price": 12.34
        }}
    ]
}}
"""

RAG_PROMPT = """
Extract structured receipt data from the following HIGH-SIGNAL lines.

FOCUSED TEXT:
{filtered_text}

FULL OCR TEXT (for reference):
{ocr_text}

Instructions:
- Prioritize TOTAL, AMOUNT, BALANCE lines
- Extract all items with prices
- Ensure total matches items when possible
- Focus on accurate price extraction

Return ONLY valid JSON:
{{
    "date": "YYYY-MM-DD",
    "total": 123.45,
    "items": [
        {{
            "description": "Item description",
            "price": 12.34
        }}
    ]
}}
"""

VISION_REPARSE_PROMPT = """
Extract structured receipt data from this high-quality OCR output.

TEXT:
{ocr_text}

Instructions:
- Extract items, prices, and total accurately
- Ensure consistency between total and item sum
- Pay attention to decimal places and currency symbols
- Include all visible items with their prices

Return ONLY valid JSON:
{{
    "date": "YYYY-MM-DD",
    "total": 123.45,
    "items": [
        {{
            "description": "Item description",
            "price": 12.34
        }}
    ]
}}
"""

def get_llm_fix_prompt(validation_error: str, ocr_text: str, previous_output: dict) -> str:
    """Generate LLM self-correction prompt"""
    return LLM_FIX_PROMPT.format(
        validation_error=validation_error,
        ocr_text=ocr_text,
        previous_output=previous_output
    )

def get_rag_prompt(filtered_text: str, ocr_text: str) -> str:
    """Generate RAG-focused context prompt"""
    return RAG_PROMPT.format(
        filtered_text=filtered_text,
        ocr_text=ocr_text
    )

def get_vision_reparse_prompt(ocr_text: str) -> str:
    """Generate Vision OCR reparse prompt"""
    return VISION_REPARSE_PROMPT.format(ocr_text=ocr_text)
