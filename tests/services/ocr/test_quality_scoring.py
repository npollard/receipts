#!/usr/bin/env python3

from services.ocr.ocr_service import OCRService
import os

def test_ocr_quality_scoring():
    print('Testing OCR quality scoring function...')

    # Initialize OCR service
    ocr_service = OCRService(use_gpu=False, lang=['en'])

    # Test on sample image
    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        print(f'Testing quality scoring on: {test_image}')

        try:
            # Extract OCR text
            ocr_text = ocr_service.extract_text(test_image)
            print(f'OCR Text length: {len(ocr_text)} characters')
            print(f'OCR Text preview: {ocr_text[:200]}...')

            # Score OCR quality
            quality_score = ocr_service.score_ocr_quality(ocr_text)

            print(f'\n=== OCR QUALITY SCORE ===')
            print(f'Quality Score: {quality_score:.3f} ({quality_score*100:.1f}%)')

            # Break down the scoring components
            print(f'\n=== SCORING BREAKDOWN ===')
            length_score = ocr_service._score_text_length(ocr_text)
            price_score = ocr_service._score_price_patterns(ocr_text)
            total_score = ocr_service._score_total_keyword(ocr_text)
            word_score = ocr_service._score_word_quality(ocr_text)
            noise_penalty = ocr_service._calculate_noise_penalty(ocr_text)

            print(f'Text Length Score: {length_score:.1f}/20')
            print(f'Price Pattern Score: {price_score:.1f}/25')
            print(f'TOTAL Keyword Score: {total_score:.1f}/20')
            print(f'Word Quality Score: {word_score:.1f}/25')
            print(f'Noise Penalty: -{noise_penalty:.1f}/10')
            print(f'Total Raw Score: {length_score + price_score + total_score + word_score - noise_penalty:.1f}/100')

            # Test with different text samples
            print(f'\n=== TESTING WITH DIFFERENT TEXT SAMPLES ===')

            # Test 1: Empty text
            empty_score = ocr_service.score_ocr_quality("")
            print(f'Empty text score: {empty_score:.3f}')

            # Test 2: Perfect receipt text
            perfect_text = "GROCERY STORE RECEIPT\nAPPLES 3.99\nORANGES 2.50\nTOTAL 6.49\nCASH 6.49\nTHANK YOU"
            perfect_score = ocr_service.score_ocr_quality(perfect_text)
            print(f'Perfect text score: {perfect_score:.3f}')

            # Test 3: Noisy text
            noisy_text = "@@@ ### !!! 123 456 789 *** &&& %%% ###"
            noisy_score = ocr_service.score_ocr_quality(noisy_text)
            print(f'Noisy text score: {noisy_score:.3f}')

            # Test 4: Short text
            short_text = "TOTAL 5.99"
            short_score = ocr_service.score_ocr_quality(short_text)
            print(f'Short text score: {short_score:.3f}')

        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('Test image not found')

if __name__ == '__main__':
    test_ocr_quality_scoring()
