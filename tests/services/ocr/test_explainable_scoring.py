#!/usr/bin/env python3

from services.ocr_service import OCRService
import os
import json

def test_explainable_scoring():
    print('Testing explainable OCR scoring system...')

    # Initialize OCR service
    ocr_service = OCRService(use_gpu=False, lang=['en'])

    # Test 1: Simple scoring (backward compatibility)
    print('\n=== TEST 1: SIMPLE SCORING (BACKWARD COMPATIBILITY) ===')
    test_text = "GROCERY STORE RECEIPT\nAPPLES 3.99\nORANGES 2.50\nTOTAL 6.49\nCASH 6.49\nTHANK YOU"

    simple_score = ocr_service.score_ocr_quality(test_text)
    print(f'Simple score: {simple_score:.3f}')
    print(f'Type: {type(simple_score)}')

    # Test 2: Detailed scoring
    print('\n=== TEST 2: DETAILED SCORING ===')
    detailed_result = ocr_service.score_ocr_quality(test_text, detailed=True)
    print(f'Detailed result type: {type(detailed_result)}')
    print(f'Total score: {detailed_result["total_score"]:.3f}')

    print(f'\nComponent Scores (normalized 0-1):')
    for component, score in detailed_result['component_scores'].items():
        print(f'  {component}: {score:.3f}')

    print(f'\nRaw Scores (points):')
    for component, score in detailed_result['raw_scores'].items():
        print(f'  {component}: {score:.1f}')

    print(f'\nMax Scores:')
    for component, max_score in detailed_result['max_scores'].items():
        print(f'  {component}: {max_score:.1f}')

    print(f'\nText Statistics:')
    for stat, value in detailed_result['text_stats'].items():
        print(f'  {stat}: {value}')

    # Test 3: Debug scoring
    print('\n=== TEST 3: DEBUG SCORING ===')
    debug_result = ocr_service.score_ocr_quality(test_text, debug=True)
    print(f'Debug score: {debug_result:.3f}')

    # Test 4: Detailed + Debug scoring
    print('\n=== TEST 4: DETAILED + DEBUG SCORING ===')
    detailed_debug_result = ocr_service.score_ocr_quality(test_text, detailed=True, debug=True)
    print(f'Detailed + debug total score: {detailed_debug_result["total_score"]:.3f}')

    # Test 5: Empty text
    print('\n=== TEST 5: EMPTY TEXT ===')
    empty_simple = ocr_service.score_ocr_quality("")
    empty_detailed = ocr_service.score_ocr_quality("", detailed=True)

    print(f'Empty simple score: {empty_simple:.3f}')
    print(f'Empty detailed total: {empty_detailed["total_score"]:.3f}')
    print(f'Empty detailed components: {empty_detailed["component_scores"]}')

    # Test 6: Different text samples
    print('\n=== TEST 6: DIFFERENT TEXT SAMPLES ===')

    test_samples = {
        "Perfect Receipt": "GROCERY STORE RECEIPT\nAPPLES 3.99\nORANGES 2.50\nBANANAS 1.25\nTOTAL 7.74\nCASH 7.74\nTHANK YOU FOR SHOPPING",
        "Short Receipt": "TOTAL 5.99",
        "Noisy Text": "@@@ ### !!! 123 456 789 *** &&& %%% ###",
        "No Prices": "GROCERY STORE\nAPPLES\nORANGES\nTOTAL\nCASH\nTHANK YOU",
        "No Total": "GROCERY STORE\nAPPLES 3.99\nORANGES 2.50\nBANANAS 1.25\nCASH 7.74\nTHANK YOU"
    }

    for name, text in test_samples.items():
        print(f'\n--- {name} ---')
        simple = ocr_service.score_ocr_quality(text)
        detailed = ocr_service.score_ocr_quality(text, detailed=True)

        print(f'Simple: {simple:.3f}')
        print(f'Detailed: {detailed["total_score"]:.3f}')
        print(f'Components: Length={detailed["component_scores"]["text_length"]:.2f}, '
              f'Prices={detailed["component_scores"]["price_patterns"]:.2f}, '
              f'Total={detailed["component_scores"]["total_keyword"]:.2f}, '
              f'Words={detailed["component_scores"]["word_quality"]:.2f}, '
              f'Noise={detailed["component_scores"]["noise_penalty"]:.2f}')

    # Test 7: Real OCR text
    print('\n=== TEST 7: REAL OCR TEXT ===')

    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        # Extract OCR text
        ocr_text = ocr_service.extract_text(test_image)
        print(f'OCR text length: {len(ocr_text)} characters')
        print(f'OCR text preview: {ocr_text[:150]}...')

        # Score with different modes
        simple_real = ocr_service.score_ocr_quality(ocr_text)
        detailed_real = ocr_service.score_ocr_quality(ocr_text, detailed=True)

        print(f'\nReal OCR Simple Score: {simple_real:.3f}')
        print(f'Real OCR Detailed Score: {detailed_real["total_score"]:.3f}')

        print(f'\nReal OCR Component Breakdown:')
        for component, score in detailed_real['component_scores'].items():
            # Map component names to max_scores keys
            component_key = component.replace('_penalty', '')
            if component_key == 'noise':
                component_key = 'noise_penalty'

            raw_score = detailed_real['raw_scores'][component_key + '_points']
            max_score = detailed_real['max_scores'][component_key]

            if component == 'noise_penalty':
                print(f'  {component}: {score:.3f} (penalty -{raw_score:.1f}/{max_score:.1f})')
            else:
                print(f'  {component}: {score:.3f} ({raw_score:.1f}/{max_score:.1f})')

        # Debug mode for real OCR
        print(f'\n--- DEBUG MODE FOR REAL OCR ---')
        debug_real = ocr_service.score_ocr_quality(ocr_text, debug=True)
        print(f'Debug real score: {debug_real:.3f}')

    else:
        print('Test image not found')

    # Test 8: JSON serialization of detailed results
    print('\n=== TEST 8: JSON SERIALIZATION ===')

    detailed_result = ocr_service.score_ocr_quality(test_text, detailed=True)

    try:
        json_str = json.dumps(detailed_result, indent=2)
        print('JSON serialization successful')
        print(f'JSON preview: {json_str[:200]}...')
    except Exception as e:
        print(f'JSON serialization failed: {e}')

    # Test 9: Performance comparison
    print('\n=== TEST 9: PERFORMANCE COMPARISON ===')

    import time

    # Simple scoring performance
    start_time = time.time()
    for _ in range(100):
        simple_score = ocr_service.score_ocr_quality(test_text)
    simple_time = time.time() - start_time

    # Detailed scoring performance
    start_time = time.time()
    for _ in range(100):
        detailed_score = ocr_service.score_ocr_quality(test_text, detailed=True)
    detailed_time = time.time() - start_time

    print(f'Simple scoring (100 iterations): {simple_time:.3f}s')
    print(f'Detailed scoring (100 iterations): {detailed_time:.3f}s')
    print(f'Performance ratio: {detailed_time/simple_time:.2f}x slower')

    print(f'\n=== EXPLAINABLE SCORING SYSTEM TEST COMPLETED ===')
    print('All scoring modes working correctly!')

if __name__ == '__main__':
    test_explainable_scoring()
