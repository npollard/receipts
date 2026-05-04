"""
OCR Configuration Settings

This module provides configuration management for OCR services,
including environment variable support and default values.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OCRConfig:
    """Configuration settings for OCR services"""
    
    # OCR Engine Settings
    use_gpu: bool = False
    languages: list = None
    confidence_threshold: float = 0.7
    
    # Quality and Fallback Settings
    quality_threshold: float = 0.25  # Default threshold for fallback to Vision OCR
    
    # Debug and Development Settings
    debug: bool = False
    debug_ocr: bool = False
    comparison_mode: bool = False
    
    # OpenAI Vision Settings (for fallback)
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.languages is None:
            self.languages = ['en']
        
        # Ensure thresholds are within valid ranges
        self.confidence_threshold = max(0.0, min(1.0, self.confidence_threshold))
        self.quality_threshold = max(0.0, min(1.0, self.quality_threshold))
        self.openai_temperature = max(0.0, min(2.0, self.openai_temperature))
    
    @classmethod
    def from_environment(cls) -> 'OCRConfig':
        """
        Create OCRConfig from environment variables
        
        Environment Variables:
        - OCR_USE_GPU: Enable GPU acceleration (true/false)
        - OCR_LANGUAGES: Comma-separated language codes (e.g., "en,ch")
        - OCR_CONFIDENCE_THRESHOLD: Minimum confidence score (0.0-1.0)
        - OCR_QUALITY_THRESHOLD: Quality threshold for fallback (0.0-1.0)
        - OCR_DEBUG: Enable debug logging (true/false)
        - OCR_DEBUG_OCR: Enable OCR decision debug (true/false)
        - OCR_COMPARISON_MODE: Enable comparison mode (true/false)
        - OCR_OPENAI_MODEL: OpenAI model for fallback
        - OCR_OPENAI_TEMPERATURE: OpenAI temperature (0.0-2.0)
        """
        return cls(
            use_gpu=cls._get_bool_env('OCR_USE_GPU', False),
            languages=cls._get_list_env('OCR_LANGUAGES', ['en']),
            confidence_threshold=cls._get_float_env('OCR_CONFIDENCE_THRESHOLD', 0.7),
            quality_threshold=cls._get_float_env('OCR_QUALITY_THRESHOLD', 0.25),
            debug=cls._get_bool_env('OCR_DEBUG', False),
            debug_ocr=cls._get_bool_env('OCR_DEBUG_OCR', False),
            comparison_mode=cls._get_bool_env('OCR_COMPARISON_MODE', False),
            openai_model=cls._get_str_env('OCR_OPENAI_MODEL', 'gpt-4o-mini'),
            openai_temperature=cls._get_float_env('OCR_OPENAI_TEMPERATURE', 0.0),
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OCRConfig':
        """
        Create OCRConfig from a dictionary
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            OCRConfig instance
        """
        return cls(
            use_gpu=config_dict.get('use_gpu', False),
            languages=config_dict.get('languages', ['en']),
            confidence_threshold=config_dict.get('confidence_threshold', 0.7),
            quality_threshold=config_dict.get('quality_threshold', 0.25),
            debug=config_dict.get('debug', False),
            debug_ocr=config_dict.get('debug_ocr', False),
            comparison_mode=config_dict.get('comparison_mode', False),
            openai_model=config_dict.get('openai_model', 'gpt-4o-mini'),
            openai_temperature=config_dict.get('openai_temperature', 0.0),
        )
    
    def override(self, **kwargs) -> 'OCRConfig':
        """
        Create a new OCRConfig with overridden values
        
        Args:
            **kwargs: Configuration values to override
            
        Returns:
            New OCRConfig instance with overridden values
        """
        # Create new config from current values
        new_config = OCRConfig(
            use_gpu=self.use_gpu,
            languages=self.languages.copy(),
            confidence_threshold=self.confidence_threshold,
            quality_threshold=self.quality_threshold,
            debug=self.debug,
            debug_ocr=self.debug_ocr,
            comparison_mode=self.comparison_mode,
            openai_model=self.openai_model,
            openai_temperature=self.openai_temperature,
        )
        
        # Apply overrides
        for key, value in kwargs.items():
            if key in new_config.__dict__:
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        return new_config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert OCRConfig to dictionary
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'use_gpu': self.use_gpu,
            'languages': self.languages,
            'confidence_threshold': self.confidence_threshold,
            'quality_threshold': self.quality_threshold,
            'debug': self.debug,
            'debug_ocr': self.debug_ocr,
            'comparison_mode': self.comparison_mode,
            'openai_model': self.openai_model,
            'openai_temperature': self.openai_temperature,
        }
    
    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.environ.get(key, '').lower()
        return value in ('true', '1', 'yes', 'on') if value else default
    
    @staticmethod
    def _get_float_env(key: str, default: float) -> float:
        """Get float value from environment variable"""
        try:
            value = os.environ.get(key)
            return float(value) if value else default
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _get_str_env(key: str, default: str) -> str:
        """Get string value from environment variable"""
        return os.environ.get(key, default)
    
    @staticmethod
    def _get_list_env(key: str, default: list) -> list:
        """Get list value from environment variable (comma-separated)"""
        value = os.environ.get(key)
        if value:
            return [item.strip() for item in value.split(',') if item.strip()]
        return default
    
    def __str__(self) -> str:
        """String representation of OCRConfig"""
        return f"""OCRConfig:
  GPU: {self.use_gpu}
  Languages: {self.languages}
  Confidence Threshold: {self.confidence_threshold}
  Quality Threshold: {self.quality_threshold}
  Debug: {self.debug}
  Debug OCR: {self.debug_ocr}
  Comparison Mode: {self.comparison_mode}
  OpenAI Model: {self.openai_model}
  OpenAI Temperature: {self.openai_temperature}"""
    
    def __repr__(self) -> str:
        """Detailed representation of OCRConfig"""
        return (f"OCRConfig(use_gpu={self.use_gpu}, languages={self.languages}, "
                f"confidence_threshold={self.confidence_threshold}, "
                f"quality_threshold={self.quality_threshold}, "
                f"debug={self.debug}, debug_ocr={self.debug_ocr}, "
                f"comparison_mode={self.comparison_mode}, "
                f"openai_model='{self.openai_model}', "
                f"openai_temperature={self.openai_temperature})")


# Default configuration instance
DEFAULT_OCR_CONFIG = OCRConfig()

# Environment-based configuration instance
ENV_OCR_CONFIG = OCRConfig.from_environment()
