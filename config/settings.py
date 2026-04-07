"""Application settings and configuration"""

import os
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_TEST = ENVIRONMENT == "test" or "pytest" in os.getenv("PYTEST_CURRENT_TEST", "").lower()

# Database Configuration
class DatabaseConfig:
    """Database settings based on environment"""
    
    @property
    def database_url(self) -> str:
        """Get database URL based on environment"""
        # Check for explicit DATABASE_URL first
        explicit_url = os.getenv("DATABASE_URL")
        if explicit_url:
            return explicit_url
        
        # Use environment-specific database
        if IS_TEST:
            return f"sqlite:///{BASE_DIR}/test_receipts.db"
        else:
            return f"sqlite:///{BASE_DIR}/receipts.db"
    
    @property
    def database_path(self) -> Path:
        """Get database file path"""
        if self.database_url.startswith("sqlite"):
            # Extract path from sqlite:///path
            path = self.database_url.replace("sqlite:///", "")
            return Path(path)
        else:
            # Non-SQLite databases don't have file paths
            return Path()
    
    @property
    def database_name(self) -> str:
        """Get database name for logging"""
        if IS_TEST:
            return "test_receipts.db"
        else:
            return "receipts.db"


# Application Configuration
class AppConfig:
    """General application settings"""
    
    # Environment
    ENVIRONMENT = ENVIRONMENT
    IS_TEST = IS_TEST
    IS_DEVELOPMENT = ENVIRONMENT == "development"
    IS_PRODUCTION = ENVIRONMENT == "production"
    
    # Paths
    BASE_DIR = BASE_DIR
    IMAGES_DIR = BASE_DIR / "imgs"
    
    # Database
    database = DatabaseConfig()
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if IS_PRODUCTION else "DEBUG")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Processing Settings
    DEFAULT_USER_EMAIL = os.getenv("DEFAULT_USER_EMAIL", "user@receipts.local")
    MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    
    # Token Usage
    TOKEN_USAGE_FILE = os.getenv("TOKEN_USAGE_FILE", "token_usage.json")
    
    @classmethod
    def get_environment_summary(cls) -> str:
        """Get summary of current environment"""
        return f"""
Environment Configuration:
=========================
Environment: {cls.ENVIRONMENT}
Is Test: {cls.IS_TEST}
Database: {cls.database.database_url}
Database Path: {cls.database.database_path}
Log Level: {cls.LOG_LEVEL}
OpenAI Model: {cls.OPENAI_MODEL}
Images Dir: {cls.IMAGES_DIR}
        """.strip()


# Global instances
db_config = DatabaseConfig()
app_config = AppConfig()

# Export commonly used settings
DATABASE_URL = db_config.database_url
DATABASE_PATH = db_config.database_path
DATABASE_NAME = db_config.database_name
ENVIRONMENT = app_config.ENVIRONMENT
IS_TEST = app_config.IS_TEST
LOG_LEVEL = app_config.LOG_LEVEL
OPENAI_MODEL = app_config.OPENAI_MODEL
DEFAULT_USER_EMAIL = app_config.DEFAULT_USER_EMAIL
