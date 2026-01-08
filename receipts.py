import os
import cv2
import json
import pytesseract
from pathlib import Path
from PIL import Image
from openai import OpenAI

ai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    return img

def parse_receipt(ocr_text):
    prompt = f"""
You are a receipt parser.

Given the OCR text below, extract:
- Date
- Items (name, category, price paid)
- Total

Return valid JSON only.

OCR TEXT:
{ocr_text}
"""

    response = ai_client.responses.create(
        model="gpt-5-nano",
        instructions=prompt,
        input=ocr_text
    )

    return response.output_text


def process_img(filename):
    print('\n\n****************')
    print('****************')
    print('FILENAME: ', filename)
    img = preprocess_image(filename)
    raw_text = pytesseract.image_to_string(img)
    print(raw_text)
    print('****************')
    receipt = parse_receipt(raw_text)
    print(json.dumps(receipt, indent=2))


# for filename in ['test.png', 'screenshot.png']:
#     process_img(filename)

filenames = Path('./imgs').rglob('*.jpg')
for filename in filenames:
    process_img(str(filename))
