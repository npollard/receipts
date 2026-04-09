"""Configuration module"""

from .settings import (
    DatabaseConfig,
    AppConfig,
    db_config,
    app_config,
    DATABASE_URL,
    DATABASE_PATH,
    DATABASE_NAME,
    ENVIRONMENT,
    IS_TEST,
    LOG_LEVEL,
    OPENAI_MODEL,
    DEFAULT_USER_EMAIL
)
from .runtime_config import (
    RuntimeConfig,
    ExecutionMode,
    get_runtime_config,
    set_runtime_config,
    reset_runtime_config,
    enforce_thread_limits,
    create_config_from_env,
)

__all__ = [
    'DatabaseConfig',
    'AppConfig',
    'db_config',
    'app_config',
    'DATABASE_URL',
    'DATABASE_PATH',
    'DATABASE_NAME',
    'ENVIRONMENT',
    'IS_TEST',
    'LOG_LEVEL',
    'OPENAI_MODEL',
    'DEFAULT_USER_EMAIL',
    # Runtime config exports
    'RuntimeConfig',
    'ExecutionMode',
    'get_runtime_config',
    'set_runtime_config',
    'reset_runtime_config',
    'enforce_thread_limits',
    'create_config_from_env',
]
