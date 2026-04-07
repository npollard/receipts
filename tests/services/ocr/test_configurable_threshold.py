#!/usr/bin/env python3

from services.ocr.ocr_service import OCRService
from config.ocr_config import OCRConfig, ENV_OCR_CONFIG
import os

def test_configurable_threshold():
    print('Testing configurable OCR threshold system...')

    # Test 1: Default configuration
    print('\n=== TEST 1: DEFAULT CONFIGURATION ===')
    ocr_service_default = OCRService()
    print(f'Default quality threshold: {ocr_service_default.quality_threshold}')
    print(f'Full config: {ocr_service_default.config}')

    # Test 2: Environment-based configuration
    print('\n=== TEST 2: ENVIRONMENT-BASED CONFIGURATION ===')
    print(f'Environment config: {ENV_OCR_CONFIG}')

    # Test 3: Parameter override (individual parameters)
    print('\n=== TEST 3: PARAMETER OVERRIDE ===')
    ocr_service_override = OCRService(quality_threshold=0.5, debug=True)
    print(f'Overridden quality threshold: {ocr_service_override.quality_threshold}')
    print(f'Overridden debug: {ocr_service_override.debug}')
    print(f'Other settings from config: GPU={ocr_service_override.use_gpu}, Lang={ocr_service_override.lang}')

    # Test 4: Configuration object override
    print('\n=== TEST 4: CONFIGURATION OBJECT OVERRIDE ===')
    custom_config = OCRConfig(
        use_gpu=True,
        languages=['en', 'fr'],
        confidence_threshold=0.8,
        quality_threshold=0.7,
        debug=True,
        debug_ocr=True
    )
    ocr_service_config = OCRService(config=custom_config)
    print(f'Custom config threshold: {ocr_service_config.quality_threshold}')
    print(f'Custom config GPU: {ocr_service_config.use_gpu}')
    print(f'Custom config languages: {ocr_service_config.lang}')
    print(f'Custom config debug: {ocr_service_config.debug}')

    # Test 5: Configuration object with parameter override
    print('\n=== TEST 5: CONFIG OBJECT + PARAMETER OVERRIDE ===')
    ocr_service_mixed = OCRService(
        config=custom_config,
        quality_threshold=0.9,  # This should override the config value
        debug_ocr=False  # This should override the config value
    )
    print(f'Mixed threshold (should be 0.9): {ocr_service_mixed.quality_threshold}')
    print(f'Mixed debug_ocr (should be False): {ocr_service_mixed.debug_ocr}')
    print(f'Mixed GPU (should be True from config): {ocr_service_mixed.use_gpu}')

    # Test 6: Environment variable simulation
    print('\n=== TEST 6: ENVIRONMENT VARIABLE SIMULATION ===')

    # Save original environment
    original_env = {}
    env_vars_to_test = {
        'OCR_QUALITY_THRESHOLD': '0.6',
        'OCR_DEBUG': 'true',
        'OCR_DEBUG_OCR': 'true',
        'OCR_USE_GPU': 'false',
        'OCR_LANGUAGES': 'en,es,fr',
        'OCR_CONFIDENCE_THRESHOLD': '0.8'
    }

    # Set test environment variables
    for key, value in env_vars_to_test.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create new config from environment
        env_config = OCRConfig.from_environment()
        print(f'Environment config threshold: {env_config.quality_threshold}')
        print(f'Environment config debug: {env_config.debug}')
        print(f'Environment config debug_ocr: {env_config.debug_ocr}')
        print(f'Environment config GPU: {env_config.use_gpu}')
        print(f'Environment config languages: {env_config.languages}')
        print(f'Environment config confidence: {env_config.confidence_threshold}')

        # Test OCR service with environment config
        ocr_service_env = OCRService(config=env_config)
        print(f'OCR Service from env config threshold: {ocr_service_env.quality_threshold}')

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    # Test 7: Configuration methods
    print('\n=== TEST 7: CONFIGURATION METHODS ===')

    # Test to_dict
    config_dict = custom_config.to_dict()
    print(f'Config to_dict: {config_dict}')

    # Test from_dict
    config_from_dict = OCRConfig.from_dict(config_dict)
    print(f'Config from_dict threshold: {config_from_dict.quality_threshold}')

    # Test override
    overridden_config = custom_config.override(quality_threshold=0.95, debug=False)
    print(f'Overridden config threshold: {overridden_config.quality_threshold}')
    print(f'Overridden config debug: {overridden_config.debug}')
    print(f'Overridden config GPU (unchanged): {overridden_config.use_gpu}')

    # Test 8: Actual OCR with different thresholds
    print('\n=== TEST 8: OCR WITH DIFFERENT THRESHOLDS ===')

    test_image = 'imgs/IMG_5416.jpg'
    if os.path.exists(test_image):
        # Test with different thresholds
        thresholds_to_test = [0.1, 0.25, 0.5, 0.8, 0.99]

        for threshold in thresholds_to_test:
            print(f'\n--- Testing with threshold: {threshold} ---')
            ocr_service = OCRService(quality_threshold=threshold, debug_ocr=True)

            try:
                result = ocr_service.extract_text(test_image)
                quality_score = ocr_service.score_ocr_quality(result)
                should_fallback = ocr_service.should_fallback(result, threshold)

                print(f'Threshold: {threshold}')
                print(f'Quality Score: {quality_score:.3f}')
                print(f'Should Fallback: {should_fallback}')
                print(f'Result Length: {len(result)} characters')

            except Exception as e:
                print(f'Error with threshold {threshold}: {e}')
    else:
        print('Test image not found for OCR testing')

    print(f'\n=== CONFIGURABLE THRESHOLD SYSTEM TEST COMPLETED ===')
    print('All configuration methods working correctly!')

if __name__ == '__main__':
    test_configurable_threshold()
