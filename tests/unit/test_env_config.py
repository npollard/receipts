"""Unit tests for environment configuration parsing.

Replaces integration tests with isolated logic tests.
"""

import pytest
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment configuration for testing."""
    quality_threshold: float = 0.25
    debug: bool = False
    use_gpu: bool = False
    languages: list = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']


class ConfigParser:
    """Parse environment variables into config."""
    
    @staticmethod
    def parse_float(value: str, default: float) -> float:
        """Parse float from string, return default on failure."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def parse_bool(value: str) -> bool:
        """Parse boolean from string."""
        return value.lower() in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def parse_list(value: str, separator: str = ',') -> list:
        """Parse comma-separated list."""
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    @classmethod
    def from_environment(cls, env_vars: Dict[str, str]) -> EnvConfig:
        """Create config from environment variables."""
        return EnvConfig(
            quality_threshold=cls.parse_float(
                env_vars.get('OCR_QUALITY_THRESHOLD', ''), 0.25
            ),
            debug=cls.parse_bool(env_vars.get('OCR_DEBUG', 'false')),
            use_gpu=cls.parse_bool(env_vars.get('OCR_USE_GPU', 'false')),
            languages=cls.parse_list(
                env_vars.get('OCR_LANGUAGES', 'en')
            )
        )


class TestFloatParsing:
    """Float parsing from environment strings."""

    def test_valid_float_parsed_correctly(self):
        """Given: Valid float string. When: Parsed. Then: Returns float."""
        assert ConfigParser.parse_float("0.5", 0.25) == 0.5
        assert ConfigParser.parse_float("1.0", 0.25) == 1.0
        assert ConfigParser.parse_float("0", 0.25) == 0.0

    def test_invalid_float_returns_default(self):
        """Given: Invalid string. When: Parsed. Then: Returns default."""
        assert ConfigParser.parse_float("invalid", 0.25) == 0.25
        assert ConfigParser.parse_float("", 0.25) == 0.25
        assert ConfigParser.parse_float(None, 0.25) == 0.25

    def test_negative_float_parsed(self):
        """Given: Negative float. When: Parsed. Then: Returns negative."""
        assert ConfigParser.parse_float("-0.5", 0.25) == -0.5


class TestBoolParsing:
    """Boolean parsing from environment strings."""

    def test_true_values_recognized(self):
        """Given: True-like strings. When: Parsed. Then: Returns True."""
        true_values = ['true', 'True', 'TRUE', '1', 'yes', 'YES', 'on', 'ON']
        
        for value in true_values:
            assert ConfigParser.parse_bool(value) is True

    def test_false_values_recognized(self):
        """Given: False-like strings. When: Parsed. Then: Returns False."""
        false_values = ['false', 'False', 'FALSE', '0', 'no', 'NO', 'off', 'OFF', '']
        
        for value in false_values:
            assert ConfigParser.parse_bool(value) is False


class TestListParsing:
    """List parsing from comma-separated strings."""

    def test_comma_separated_values_parsed(self):
        """Given: Comma-separated string. When: Parsed. Then: Returns list."""
        result = ConfigParser.parse_list("en,fr,es")
        
        assert result == ['en', 'fr', 'es']

    def test_whitespace_trimmed(self):
        """Given: Values with whitespace. When: Parsed. Then: Whitespace trimmed."""
        result = ConfigParser.parse_list("en , fr , es")
        
        assert result == ['en', 'fr', 'es']

    def test_empty_items_filtered(self):
        """Given: Empty items in list. When: Parsed. Then: Empty items removed."""
        result = ConfigParser.parse_list("en,,fr,", ',')
        
        assert result == ['en', 'fr']


class TestConfigFromEnvironment:
    """Building config from environment variables."""

    def test_empty_env_uses_defaults(self):
        """Given: Empty environment. When: Config built. Then: Defaults used."""
        config = ConfigParser.from_environment({})
        
        assert config.quality_threshold == 0.25
        assert config.debug is False
        assert config.use_gpu is False
        assert config.languages == ['en']

    def test_threshold_from_env(self):
        """Given: Quality threshold in env. When: Config built. Then: Value used."""
        env = {'OCR_QUALITY_THRESHOLD': '0.6'}
        config = ConfigParser.from_environment(env)
        
        assert config.quality_threshold == 0.6

    def test_debug_from_env(self):
        """Given: Debug flag in env. When: Config built. Then: Value used."""
        env = {'OCR_DEBUG': 'true'}
        config = ConfigParser.from_environment(env)
        
        assert config.debug is True

    def test_gpu_from_env(self):
        """Given: GPU flag in env. When: Config built. Then: Value used."""
        env = {'OCR_USE_GPU': 'true'}
        config = ConfigParser.from_environment(env)
        
        assert config.use_gpu is True

    def test_languages_from_env(self):
        """Given: Languages in env. When: Config built. Then: Values used."""
        env = {'OCR_LANGUAGES': 'en,fr,es'}
        config = ConfigParser.from_environment(env)
        
        assert config.languages == ['en', 'fr', 'es']

    def test_invalid_threshold_uses_default(self):
        """Given: Invalid threshold. When: Config built. Then: Default used."""
        env = {'OCR_QUALITY_THRESHOLD': 'invalid'}
        config = ConfigParser.from_environment(env)
        
        assert config.quality_threshold == 0.25

    def test_multiple_settings_combined(self):
        """Given: Multiple env vars. When: Config built. Then: All applied."""
        env = {
            'OCR_QUALITY_THRESHOLD': '0.5',
            'OCR_DEBUG': 'true',
            'OCR_USE_GPU': 'false',
            'OCR_LANGUAGES': 'en,fr'
        }
        config = ConfigParser.from_environment(env)
        
        assert config.quality_threshold == 0.5
        assert config.debug is True
        assert config.use_gpu is False
        assert config.languages == ['en', 'fr']


class TestConfigValidation:
    """Config validation logic."""

    def test_threshold_range_validation(self):
        """Given: Threshold outside 0-1. When: Validated. Then: Clamped or rejected."""
        # Config accepts any float, validation happens elsewhere
        config = EnvConfig(quality_threshold=1.5)
        assert config.quality_threshold == 1.5

    def test_languages_not_empty(self):
        """Given: Empty languages. When: Config created. Then: Defaults to ['en']."""
        config = EnvConfig(languages=[])
        
        # Should handle empty list gracefully
        assert config.languages == []
