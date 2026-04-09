"""Centralized runtime configuration for concurrency control

This module provides a single source of truth for all concurrency-related settings.
It enforces that there is exactly ONE layer controlling parallelism.
"""

import os
import sys
from typing import Literal, Optional
from dataclasses import dataclass
from enum import Enum


class ExecutionMode(Enum):
    """Execution modes with different parallelism profiles"""
    DEV = "dev"           # Safe, single-threaded for development
    LOCAL = "local"       # Limited parallelism for local batch processing
    CLOUD = "cloud"       # Scalable parallelism for cloud deployment


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration for concurrency control
    
    This is the SINGLE source of truth for all parallelism settings.
    All services must derive their concurrency settings from this config.
    
    Attributes:
        mode: Execution mode (dev/local/cloud)
        max_workers: Maximum number of parallel workers for batch processing
        ocr_threads: Number of threads for OCR operations
        torch_threads: Number of threads for torch operations
        omp_threads: Number of threads for OpenMP operations
        mkl_threads: Number of threads for MKL operations
        openblas_threads: Number of threads for OpenBLAS operations
    """
    mode: ExecutionMode
    max_workers: int
    ocr_threads: int
    torch_threads: int
    omp_threads: int
    mkl_threads: int
    openblas_threads: int
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Enforce no nested parallelism
        if self.max_workers > 1 and self.ocr_threads > 1:
            raise RuntimeError(
                f"Nested parallelism detected: max_workers={self.max_workers}, "
                f"ocr_threads={self.ocr_threads}. "
                f"There must be exactly ONE layer controlling parallelism. "
                f"Either set MAX_WORKERS=1 (sequential OCR in parallel batches) "
                f"or OCR_THREADS=1 (parallel OCR in sequential batches)."
            )
        
        # Validate non-negative values
        for attr in ['max_workers', 'ocr_threads', 'torch_threads', 
                     'omp_threads', 'mkl_threads', 'openblas_threads']:
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"{attr} must be non-negative, got {value}")
    
    @property
    def is_parallel(self) -> bool:
        """Check if any parallelism is enabled"""
        return self.max_workers > 1 or self.ocr_threads > 1
    
    @property
    def is_sequential(self) -> bool:
        """Check if running in sequential mode"""
        return not self.is_parallel
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary"""
        return f"""
Runtime Configuration:
======================
Mode: {self.mode.value}

Parallelism Settings:
  max_workers: {self.max_workers} (batch processing workers)
  ocr_threads: {self.ocr_threads} (OCR thread count)

Thread Limits:
  torch_threads: {self.torch_threads}
  omp_threads: {self.omp_threads}
  mkl_threads: {self.mkl_threads}
  openblas_threads: {self.openblas_threads}

Execution: {'Parallel' if self.is_parallel else 'Sequential'}
""".strip()


def get_execution_mode() -> ExecutionMode:
    """Get execution mode from environment variable"""
    mode_str = os.getenv("EXECUTION_MODE", "dev").lower()
    try:
        return ExecutionMode(mode_str)
    except ValueError:
        valid_modes = [m.value for m in ExecutionMode]
        raise ValueError(
            f"Invalid EXECUTION_MODE='{mode_str}'. "
            f"Valid modes: {', '.join(valid_modes)}"
        )


def create_config_from_env() -> RuntimeConfig:
    """Create RuntimeConfig from environment variables
    
    Environment variables:
        EXECUTION_MODE: dev, local, or cloud (default: dev)
        MAX_WORKERS: Number of parallel workers (default: mode-dependent)
        OCR_THREADS: Number of OCR threads (default: mode-dependent)
    """
    mode = get_execution_mode()
    
    # Mode-dependent defaults
    defaults = {
        ExecutionMode.DEV: {
            'max_workers': 1,
            'ocr_threads': 1,
        },
        ExecutionMode.LOCAL: {
            'max_workers': 2,
            'ocr_threads': 1,
        },
        ExecutionMode.CLOUD: {
            'max_workers': 4,
            'ocr_threads': 1,
        },
    }
    
    mode_defaults = defaults[mode]
    
    # Read from environment with mode-specific defaults
    max_workers = int(os.getenv("MAX_WORKERS", mode_defaults['max_workers']))
    ocr_threads = int(os.getenv("OCR_THREADS", mode_defaults['ocr_threads']))
    
    # Thread limits: default to ocr_threads for consistency
    torch_threads = int(os.getenv("TORCH_THREADS", ocr_threads))
    omp_threads = int(os.getenv("OMP_NUM_THREADS", ocr_threads))
    mkl_threads = int(os.getenv("MKL_NUM_THREADS", ocr_threads))
    openblas_threads = int(os.getenv("OPENBLAS_NUM_THREADS", ocr_threads))
    
    return RuntimeConfig(
        mode=mode,
        max_workers=max_workers,
        ocr_threads=ocr_threads,
        torch_threads=torch_threads,
        omp_threads=omp_threads,
        mkl_threads=mkl_threads,
        openblas_threads=openblas_threads,
    )


# Global runtime config instance (lazy-loaded)
_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """Get the global runtime configuration
    
    This is the primary entry point for accessing runtime config.
    The config is created on first access.
    """
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = create_config_from_env()
    return _runtime_config


def set_runtime_config(config: RuntimeConfig) -> None:
    """Set the global runtime configuration explicitly
    
    Useful for testing or when config needs to be set programmatically.
    """
    global _runtime_config
    _runtime_config = config


def reset_runtime_config() -> None:
    """Reset the global runtime configuration
    
    Forces recreation on next access. Useful for testing.
    """
    global _runtime_config
    _runtime_config = None


def enforce_thread_limits(config: Optional[RuntimeConfig] = None) -> None:
    """Enforce thread limits at application entrypoint
    
    This MUST be called BEFORE importing torch, numpy, or easyocr.
    It sets environment variables and configures torch to prevent
    hidden parallelism from consuming all CPU cores.
    
    Args:
        config: RuntimeConfig to use (defaults to global config)
    """
    if config is None:
        config = get_runtime_config()
    
    # Set environment variables BEFORE any library imports
    os.environ["OMP_NUM_THREADS"] = str(config.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.mkl_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(config.openblas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(config.openblas_threads)  # macOS
    os.environ["NUMEXPR_NUM_THREADS"] = str(config.omp_threads)
    
    # Configure torch if already imported, otherwise it will respect env vars
    if "torch" in sys.modules:
        import torch
        torch.set_num_threads(config.torch_threads)
        torch.set_num_interop_threads(1)  # Prevent interop parallelism
    
    # Log enforcement
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Thread limits enforced: OMP={config.omp_threads}, "
        f"MKL={config.mkl_threads}, OPENBLAS={config.openblas_threads}, "
        f"TORCH={config.torch_threads}"
    )


# Convenience exports
__all__ = [
    'RuntimeConfig',
    'ExecutionMode',
    'get_runtime_config',
    'set_runtime_config',
    'reset_runtime_config',
    'enforce_thread_limits',
    'create_config_from_env',
]
