# Architecture Deep Dive

## Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                       │
│  (CLI: main.py, API endpoints - not yet implemented)   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                    │
│  (BatchProcessingService - orchestrates workflow)        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     Domain Layer                       │
│  (ReceiptParser, ValidationService - business logic)      │
│  No dependencies on infrastructure or services          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                  │
│  (OCRService, Vision OCR, Database)                     │
└─────────────────────────────────────────────────────────┘
```

## Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Receipt   │───→│  EasyOCR     │───→│  GPT-4o-mini │───→│   Pydantic   │
│   Image     │    │  (local)     │    │   Parser     │    │ Validation   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                        │                                            │
                        │ (low quality)                              │ (fails)
                        ▼                                            │
              ┌─────────────────┐                                    │
              │ OpenAI Vision   │◄───────────────────────────────────┘
              │ (fallback)      │          (retry with corrections)
              └─────────────────┘
```

## Key Design Decisions

- **Interface-driven design**: `ImageProcessingInterface` and `ReceiptParsingInterface` enforce contracts, enable DI and testability
- **Clean architecture**: Domain layer has no dependencies on infrastructure or services
- **Single source of truth**: No `hasattr` branching; interfaces define explicit methods
- **Local OCR first**: EasyOCR runs locally (free, fast), falls back to OpenAI Vision only when quality is low
- **Lazy initialization**: OCRService is only instantiated when needed, preventing heavy imports during tests
- **Structured validation**: Pydantic v2 models enforce schema + data types
- **Retry strategies adapt to error severity**: Small errors → self-correction; Large errors → RAG or OCR reprocessing
- **Always preserve parsed data**: Even failed validations return partial results for debugging
- **Deterministic OCR scoring**: Structured scoring with clear thresholds (bad ≤0.4, medium ≥0.3, good ≥0.5)
- **Single-layer parallelism**: BatchProcessingService is the ONLY layer controlling concurrency
- **Thread limits enforced at entrypoint**: OMP, MKL, OpenBLAS, and torch threads are configured before any heavy imports
