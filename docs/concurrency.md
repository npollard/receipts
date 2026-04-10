# Concurrency Model

## Single-Layer Parallelism Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Application Entrypoint (main.py)                            │
│  ├── Enforces thread limits BEFORE any imports               │
│  ├── OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS │
│  └── torch.set_num_threads()                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  BatchProcessingService (SINGLE parallelism layer)           │
│  ├── ProcessPoolExecutor(max_workers=MAX_WORKERS)            │
│  ├── Each worker: fresh OCR + Parser instances              │
│  └── No shared stateful objects across processes            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker Process                                             │
│  ├── OCRService (OCR_THREADS=1 when parallel)              │
│  ├── ReceiptParser                                          │
│  └── Sequential processing within worker                    │
└─────────────────────────────────────────────────────────────┘
```

## Execution Modes

| Mode | MAX_WORKERS | OCR_THREADS | Use Case |
|------|-------------|-------------|----------|
| **dev** (default) | 1 | 1 | Safe development, debugging, testing |
| **local** | 2 | 1 | Local batch processing on laptops |
| **cloud** | 4 | 1 | Cloud deployment with more resources |

## Environment Variables

```bash
# Execution mode (dev/local/cloud)
export EXECUTION_MODE=local

# Override defaults (optional)
export MAX_WORKERS=2        # Number of parallel workers
export OCR_THREADS=1        # Must be 1 when MAX_WORKERS > 1
```

## Runtime Guard

The application enforces **no nested parallelism**:
- If `MAX_WORKERS > 1` AND `OCR_THREADS > 1` → RuntimeError
- This ensures exactly ONE layer controls parallelism

## Performance Characteristics

- **Default mode (dev)**: ~100% CPU (single core), deterministic, safe for laptops
- **Local mode**: ~150-200% CPU (2 workers), good throughput without saturation
- **Cloud mode**: Scales based on available cores, ~400% CPU with 4 workers
