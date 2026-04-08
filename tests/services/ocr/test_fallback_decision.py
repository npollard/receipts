#!/usr/bin/env python3

from services.ocr_service import OCRService
import os
import json

def test_fallback_decision():
    print('Testing OCR fallback decision function...')

    # Initialize OCR service
    ocr_service = OCRService(use_gpu=False, lang=['en'])

    # Test on sample image
    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        print(f'Testing fallback decision on: {test_image}')

        try:
            # Extract OCR text
            ocr_text = ocr_service.extract_text(test_image)
            print(f'OCR Text length: {len(ocr_text)} characters')
            print(f'OCR Text preview: {ocr_text[:150]}...')

            # Test fallback decision with default threshold
            print(f'\n=== FALLBACK DECISION (DEFAULT THRESHOLD) ===')
            should_fallback = ocr_service.should_fallback(ocr_text)
            print(f'Should fallback: {should_fallback}')

            # Test with different thresholds
            thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
            print(f'\n=== FALLBACK DECISION WITH DIFFERENT THRESHOLDS ===')
            for threshold in thresholds:
                fallback = ocr_service.should_fallback(ocr_text, threshold)
                print(f'Threshold {threshold:.2f}: {"FALLBACK" if fallback else "USE_EASYOCR"}')

            # Get detailed reasoning
            print(f'\n=== DETAILED REASONING ===')
            reasoning = ocr_service.get_fallback_reasoning(ocr_text)
            print(f'Recommendation: {reasoning["recommendation"]}')
            print(f'Quality Score: {reasoning["quality_score"]:.3f}')
            print(f'Threshold: {reasoning["threshold"]:.3f}')
            print(f'Reasoning: {reasoning["reasoning"]}')

            print(f'\nComponent Scores:')
            for component, score in reasoning["component_scores"].items():
                if component == "noise_penalty":
                    print(f'  {component}: -{score:.1f}')
                else:
                    print(f'  {component}: {score:.1f}')

            # Test with different text samples
            print(f'\n=== TESTING WITH DIFFERENT TEXT SAMPLES ===')

            # Test 1: Empty text
            print(f'Empty text:')
            empty_fallback = ocr_service.should_fallback("")
            empty_reasoning = ocr_service.get_fallback_reasoning("")
            print(f'  Should fallback: {empty_fallback}')
            print(f'  Reasoning: {empty_reasoning["reasoning"]}')

            # Test 2: Perfect receipt text
            perfect_text = "GROCERY STORE RECEIPT\nAPPLES 3.99\nORANGES 2.50\nTOTAL 6.49\nCASH 6.49\nTHANK YOU"
            print(f'\nPerfect receipt text:')
            perfect_fallback = ocr_service.should_fallback(perfect_text, 0.5)
            perfect_reasoning = ocr_service.get_fallback_reasoning(perfect_text, 0.5)
            print(f'  Should fallback: {perfect_fallback}')
            print(f'  Quality Score: {perfect_reasoning["quality_score"]:.3f}')
            print(f'  Reasoning: {perfect_reasoning["reasoning"]}')

            # Test 3: Noisy text
            noisy_text = "@@@ ### !!! 123 456 789 *** &&& %%% ###"
            print(f'\nNoisy text:')
            noisy_fallback = ocr_service.should_fallback(noisy_text, 0.5)
            noisy_reasoning = ocr_service.get_fallback_reasoning(noisy_text, 0.5)
            print(f'  Should fallback: {noisy_fallback}')
            print(f'  Quality Score: {noisy_reasoning["quality_score"]:.3f}')
            print(f'  Reasoning: {noisy_reasoning["reasoning"]}')

            # Test 4: Short text
            short_text = "TOTAL 5.99"
            print(f'\nShort text:')
            short_fallback = ocr_service.should_fallback(short_text, 0.5)
            short_reasoning = ocr_service.get_fallback_reasoning(short_text, 0.5)
            print(f'  Should fallback: {short_fallback}')
            print(f'  Quality Score: {short_reasoning["quality_score"]:.3f}')
            print(f'  Reasoning: {short_reasoning["reasoning"]}')

            # Test 5: Medium quality text
            medium_text = "STORE\nITEM1 4.99\nITEM2 2.50\nTOTAL 7.49\nTHANKS"
            print(f'\nMedium quality text:')
            medium_fallback = ocr_service.should_fallback(medium_text, 0.5)
            medium_reasoning = ocr_service.get_fallback_reasoning(medium_text, 0.5)
            print(f'  Should fallback: {medium_fallback}')
            print(f'  Quality Score: {medium_reasoning["quality_score"]:.3f}')
            print(f'  Reasoning: {medium_reasoning["reasoning"]}')

        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('Test image not found')

if __name__ == '__main__':
    test_fallback_decision()
