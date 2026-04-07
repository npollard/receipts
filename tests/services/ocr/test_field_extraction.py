#!/usr/bin/env python3

from services.ocr.ocr_service import OCRService
import os
import json

def test_field_extraction():
    print('Testing OCR field extraction...')

    # Initialize OCR service
    ocr_service = OCRService(use_gpu=False, lang=['en'])

    # Test on sample image
    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        print(f'Testing field extraction on: {test_image}')

        try:
            # Extract OCR text
            ocr_text = ocr_service.extract_text(test_image)
            print(f'OCR Text (first 200 chars):')
            print(ocr_text[:200] + '...')

            # Extract structured fields
            fields = ocr_service.extract_receipt_fields(ocr_text)

            print(f'\n=== STRUCTURED FIELD EXTRACTION ===')
            print(f'Total Amount: {fields["total_amount"]}')
            print(f'Merchant Name: {fields["merchant_name"]}')
            print(f'Date: {fields["date"]}')
            print(f'Items Found: {len(fields["items"])}')
            print(f'Extraction Confidence: {fields["extraction_confidence"]:.2f}')

            print(f'\n=== EXTRACTED ITEMS ===')
            for i, item in enumerate(fields['items'][:5]):
                print(f'{i+1}. {item["description"]} - ${item["price"]:.2f} (qty: {item["quantity"]})')

            if len(fields['items']) > 5:
                print(f'... and {len(fields["items"]) - 5} more items')

            print(f'\n=== FULL STRUCTURED DATA ===')
            print(json.dumps(fields, indent=2, default=str))

        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('Test image not found')

if __name__ == '__main__':
    test_field_extraction()
