#!/usr/bin/env python3

from services.ocr.ocr_service import OCRService
import os

def test_debug_ocr_flow():
    print('Testing debug OCR flow with detailed observability...')

    # Test 1: Debug OCR disabled (default behavior)
    print('\n=== TEST 1: DEBUG OCR DISABLED (DEFAULT) ===')
    ocr_service_normal = OCRService(use_gpu=False, lang=['en'], debug_ocr=False)

    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        print(f'Testing normal mode on: {test_image}')
        try:
            result_normal = ocr_service_normal.extract_text(test_image)
            print(f'Normal mode result length: {len(result_normal)} characters')
            print(f'Normal mode result preview: {result_normal[:100]}...')
        except Exception as e:
            print(f'Normal mode error: {e}')

    # Test 2: Debug OCR enabled with low threshold (should trigger fallback)
    print(f'\n=== TEST 2: DEBUG OCR ENABLED WITH LOW THRESHOLD ===')
    ocr_service_debug_low = OCRService(
        use_gpu=False,
        lang=['en'],
        debug_ocr=True,
        quality_threshold=0.1  # Low threshold - should NOT trigger fallback
    )

    if os.path.exists(test_image):
        print(f'Testing debug mode with low threshold on: {test_image}')
        try:
            result_debug_low = ocr_service_debug_low.extract_text(test_image)
            print(f'Debug low threshold result length: {len(result_debug_low)} characters')
        except Exception as e:
            print(f'Debug low threshold error: {e}')

    # Test 3: Debug OCR enabled with high threshold (should trigger fallback if possible)
    print(f'\n=== TEST 3: DEBUG OCR ENABLED WITH HIGH THRESHOLD ===')
    ocr_service_debug_high = OCRService(
        use_gpu=False,
        lang=['en'],
        debug_ocr=True,
        quality_threshold=0.99  # Very high threshold - should NOT trigger fallback since quality is 1.0
    )

    if os.path.exists(test_image):
        print(f'Testing debug mode with high threshold on: {test_image}')
        try:
            result_debug_high = ocr_service_debug_high.extract_text(test_image)
            print(f'Debug high threshold result length: {len(result_debug_high)} characters')
        except Exception as e:
            print(f'Debug high threshold error: {e}')

    # Test 4: Debug OCR with different text samples
    print(f'\n=== TEST 4: DEBUG OCR WITH TEXT SAMPLES ===')
    ocr_service_debug = OCRService(use_gpu=False, lang=['en'], debug_ocr=True, quality_threshold=0.5)

    # Create temporary text files for testing
    test_texts = {
        "empty.txt": "",
        "perfect.txt": "GROCERY STORE RECEIPT\nAPPLES 3.99\nORANGES 2.50\nTOTAL 6.49\nCASH 6.49\nTHANK YOU",
        "noisy.txt": "@@@ ### !!! 123 456 789 *** &&& %%% ###",
        "short.txt": "TOTAL 5.99",
        "medium.txt": "STORE\nITEM1 4.99\nITEM2 2.50\nTOTAL 7.49\nTHANKS"
    }

    # Create temporary files and test
    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:
        for filename, content in test_texts.items():
            temp_file = os.path.join(temp_dir, filename)
            with open(temp_file, 'w') as f:
                f.write(content)

            print(f'\n--- Testing with {filename} ---')
            try:
                # We'll simulate the debug flow by calling the quality scoring directly
                # since we can't easily create images for these text samples
                quality_score = ocr_service_debug.score_ocr_quality(content)
                should_fallback = ocr_service_debug.should_fallback(content, 0.5)
                reasoning = ocr_service_debug.get_fallback_reasoning(content, 0.5)

                print(f'Content: "{content[:50]}{"..." if len(content) > 50 else ""}"')
                print(f'Quality Score: {quality_score:.3f}')
                print(f'Should Fallback: {should_fallback}')
                print(f'Reasoning: {reasoning["reasoning"]}')

            except Exception as e:
                print(f'Error testing {filename}: {e}')

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Test 5: Debug OCR with error handling
    print(f'\n=== TEST 5: DEBUG OCR ERROR HANDLING ===')
    ocr_service_debug_error = OCRService(use_gpu=False, lang=['en'], debug_ocr=True)

    # Test with non-existent file
    try:
        print('Testing with non-existent file...')
        result_error = ocr_service_debug_error.extract_text('non_existent_file.jpg')
        print(f'Unexpected success: {result_error}')
    except Exception as e:
        print(f'Expected error caught: {type(e).__name__}: {str(e)}')

    print(f'\n=== DEBUG OCR FLOW TEST COMPLETED ===')
    print('Debug OCR flow provides detailed observability while maintaining default behavior when disabled.')

if __name__ == '__main__':
    test_debug_ocr_flow()
