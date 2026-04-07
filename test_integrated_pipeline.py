#!/usr/bin/env python3

from services.ocr_service import OCRService
import os
import json

def test_integrated_pipeline():
    print('Testing integrated OCR pipeline with quality-based fallback...')
    
    # Test with different quality thresholds
    thresholds = [0.1, 0.25, 0.5, 0.9]
    
    for threshold in thresholds:
        print(f'\n=== TESTING WITH QUALITY THRESHOLD: {threshold} ===')
        
        # Initialize OCR service with quality threshold
        ocr_service = OCRService(
            use_gpu=False, 
            lang=['en'], 
            debug=False, 
            quality_threshold=threshold
        )
        
        # Test on sample image
        test_image = 'imgs/IMG_5416.jpg'
        if os.path.exists(test_image):
            print(f'Testing on: {test_image}')
            
            try:
                # Extract text with integrated pipeline
                result = ocr_service.extract_text(test_image)
                
                # Get quality score for the result
                quality_score = ocr_service.score_ocr_quality(result)
                
                print(f'Result length: {len(result)} characters')
                print(f'Final quality score: {quality_score:.3f}')
                print(f'Result preview: {result[:150]}...')
                
                # Check if fallback was triggered
                fallback_triggered = ocr_service.should_fallback(result, threshold)
                print(f'Fallback would be triggered: {fallback_triggered}')
                
            except Exception as e:
                print(f'Error: {e}')
        else:
            print('Test image not found')
    
    # Test with different text samples to verify fallback logic
    print(f'\n=== TESTING FALLBACK LOGIC WITH TEXT SAMPLES ===')
    
    ocr_service = OCRService(use_gpu=False, lang=['en'], quality_threshold=0.5)
    
    # Test samples
    test_samples = [
        ("Empty text", ""),
        ("Perfect receipt", "GROCERY STORE RECEIPT\nAPPLES 3.99\nORANGES 2.50\nTOTAL 6.49\nCASH 6.49\nTHANK YOU"),
        ("Noisy text", "@@@ ### !!! 123 456 789 *** &&& %%% ###"),
        ("Short text", "TOTAL 5.99"),
        ("Medium quality", "STORE\nITEM1 4.99\nITEM2 2.50\nTOTAL 7.49\nTHANKS"),
    ]
    
    for name, text in test_samples:
        print(f'\n{name}:')
        quality_score = ocr_service.score_ocr_quality(text)
        should_fallback = ocr_service.should_fallback(text, 0.5)
        print(f'  Quality Score: {quality_score:.3f}')
        print(f'  Should fallback: {should_fallback}')
        
        # Get detailed reasoning
        reasoning = ocr_service.get_fallback_reasoning(text, 0.5)
        print(f'  Reasoning: {reasoning["reasoning"]}')
    
    # Test the actual pipeline integration
    print(f'\n=== TESTING ACTUAL PIPELINE INTEGRATION ===')
    
    ocr_service = OCRService(use_gpu=False, lang=['en'], quality_threshold=0.8)  # High threshold to trigger fallback
    
    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        print(f'Testing pipeline with high threshold (0.8) on: {test_image}')
        
        try:
            # This should trigger fallback since real OCR quality is 1.0, but threshold is 0.8
            # Actually, with threshold 0.8, it should NOT trigger fallback since quality is 1.0 > 0.8
            result = ocr_service.extract_text(test_image)
            print(f'Pipeline result length: {len(result)} characters')
            print(f'Pipeline result preview: {result[:150]}...')
            
        except Exception as e:
            print(f'Pipeline error: {e}')
            import traceback
            traceback.print_exc()
    
    # Test with very high threshold to force fallback
    print(f'\n=== TESTING WITH VERY HIGH THRESHOLD TO FORCE FALLBACK ===')
    
    ocr_service_high = OCRService(use_gpu=False, lang=['en'], quality_threshold=0.99)  # Very high threshold
    
    if os.path.exists(test_image):
        print(f'Testing with very high threshold (0.99) on: {test_image}')
        
        try:
            result = ocr_service_high.extract_text(test_image)
            print(f'High threshold result length: {len(result)} characters')
            print(f'High threshold result preview: {result[:150]}...')
            
        except Exception as e:
            print(f'High threshold error: {e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_integrated_pipeline()
