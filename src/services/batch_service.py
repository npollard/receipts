"""Batch processing service with controlled concurrency

This service is the SINGLE layer controlling parallelism in the application.
It uses ProcessPoolExecutor for parallel processing while ensuring no nested
parallelism occurs (OCR threads must be 1 when using multiple workers).
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path for worker processes
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from contracts.interfaces import (
    ImageProcessingInterface,
    ReceiptParsingInterface,
    BatchProcessingInterface,
)
from tracking import TokenUsage
from api_response import APIResponse
from shared.models.processing_result import ProcessingResult
from core.file_operations import get_image_files
from core.logging import get_batch_logger
from utils.output_formatter import format_receipt_result
from domain.validation import has_meaningful_receipt_data
from config import get_runtime_config, RuntimeConfig

# Reduce logging noise - set specific loggers to WARNING level
logging.getLogger('services.ocr_service').setLevel(logging.WARNING)
logging.getLogger('domain.validation.validation_service').setLevel(logging.WARNING)
logging.getLogger('services.retry_service').setLevel(logging.WARNING)

logger = get_batch_logger(__name__)


@dataclass
class BatchObservability:
    """Observability data for batch processing"""
    total_images: int
    successful: int
    failed: int
    total_time_ms: float
    avg_time_per_image_ms: float
    total_tokens: TokenUsage
    ocr_breakdown: Dict[str, int]  # Count by OCR method

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_images': self.total_images,
            'successful': self.successful,
            'failed': self.failed,
            'total_time_ms': round(self.total_time_ms, 2),
            'avg_time_per_image_ms': round(self.avg_time_per_image_ms, 2),
            'total_tokens': self.total_tokens.to_dict(),
            'ocr_breakdown': self.ocr_breakdown,
        }


def _set_worker_thread_limits(ocr_threads: int) -> None:
    """Configure thread limits for multiprocessing workers."""
    import os
    os.environ['OMP_NUM_THREADS'] = str(ocr_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(ocr_threads)
    os.environ['MKL_NUM_THREADS'] = str(ocr_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(ocr_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(ocr_threads)
    os.environ['TORCH_NUM_THREADS'] = str(ocr_threads)


def _create_processing_services():
    """Create fresh service instances for isolated worker processes."""
    from image_processing import VisionProcessor
    from domain.parsing.receipt_parser import ReceiptParser

    image_processor = VisionProcessor()
    ocr_service = image_processor.ocr_service
    receipt_parser = ReceiptParser(ocr_service=ocr_service)
    return image_processor, ocr_service, receipt_parser


def _extract_ocr_data(image_processor, image_path: str):
    """Extract OCR text and timing data."""
    import time
    ocr_start = time.time()
    ocr_text = image_processor.extract_text(image_path)
    ocr_duration_ms = (time.time() - ocr_start) * 1000
    return ocr_text, ocr_duration_ms


def _get_observability_data(ocr_service) -> dict:
    """Extract observability data from OCR service."""
    ocr_obs = getattr(ocr_service, '_last_observability', None)
    return {
        'method': ocr_obs.method if ocr_obs else 'easyocr',
        'attempted_methods': ocr_obs.attempted_methods if ocr_obs else ['easyocr'],
        'quality_score': ocr_obs.quality_score if ocr_obs else 0.0,
    }


def _convert_parsed_data(result):
    """Convert parsing result to dict format."""
    if result.parsed is None:
        return {}
    if isinstance(result.parsed, dict):
        return result.parsed
    return result.parsed.model_dump()




def _build_processing_result(
    image_path: str,
    parsed_data: dict,
    parse_result,
    receipt_parser,
    ocr_data: dict,
    processing_time_ms: float,
    success: bool
) -> ProcessingResult:
    """Build ProcessingResult from extracted data - single source of truth."""
    from tracking import TokenUsage

    return ProcessingResult(
        image_path=image_path,
        success=success,
        parsed_data=parsed_data,
        retries=receipt_parser.get_current_retries(),
        validation_error=parse_result.error if parse_result.error else None,
        token_usage=parse_result.token_usage if parse_result.token_usage else TokenUsage(),
        processing_time_ms=processing_time_ms,
        ocr_method=ocr_data['method'],
        ocr_duration_ms=ocr_data['duration_ms'],
        ocr_attempted_methods=ocr_data['attempted_methods'],
        ocr_quality_score=ocr_data['quality_score'],
    )


def _build_error_result(image_path: str, error: Optional[Exception] = None, processing_time_ms: float = 0.0) -> ProcessingResult:
    """Build ProcessingResult for failed processing - single source of truth."""
    from tracking import TokenUsage

    return ProcessingResult(
        image_path=image_path,
        success=False,
        parsed_data={},
        retries=[],
        validation_error=str(error) if error else "Processing failed",
        token_usage=TokenUsage(),
        processing_time_ms=processing_time_ms,
        ocr_method="error",
        ocr_duration_ms=0.0,
        ocr_attempted_methods=[],
        ocr_quality_score=0.0,
    )


def process_single_image(
    image_path: str,
    execution_mode: str,
    max_workers: int,
    ocr_threads: int,
) -> ProcessingResult:
    """Process a single image with fresh service instances (worker process entry point)."""
    import time
    import logging

    logger = logging.getLogger(__name__)
    start_time = time.time()

    _set_worker_thread_limits(ocr_threads)

    try:
        image_processor, ocr_service, receipt_parser = _create_processing_services()
        ocr_text, ocr_duration_ms = _extract_ocr_data(image_processor, image_path)
        obs_data = _get_observability_data(ocr_service)
        obs_data['duration_ms'] = ocr_duration_ms

        parse_result = receipt_parser.parse_with_validation_driven_retry(ocr_text, image_path)
        parsed_data = _convert_parsed_data(parse_result)
        has_data = has_meaningful_receipt_data(parsed_data) if parse_result.valid else False

        processing_time_ms = (time.time() - start_time) * 1000
        return _build_processing_result(
            image_path, parsed_data, parse_result, receipt_parser,
            obs_data, processing_time_ms, success=has_data
        )

    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error processing {image_path}: {e}")
        return _build_error_result(image_path, e, processing_time_ms)


class BatchProcessingService(BatchProcessingInterface):
    """Service for coordinating batch processing with controlled concurrency

    This is the SINGLE layer controlling parallelism. All batch processing
    goes through this service to ensure no nested parallelism occurs.
    """

    def __init__(self, runtime_config: Optional[RuntimeConfig] = None):
        self.logger = logger
        self.config = runtime_config or get_runtime_config()
        self.logger.debug(f"BatchProcessingService initialized with {self.config.get_summary()}")

    def process_batch(
        self,
        image_files: List[Path],
        image_processor: Optional[ImageProcessingInterface] = None,
        receipt_parser: Optional[ReceiptParsingInterface] = None,
    ) -> Tuple[int, int, TokenUsage, Optional[BatchObservability]]:
        """Process multiple images with controlled concurrency

        Args:
            image_files: List of image file paths to process
            image_processor: Optional custom image processor (for testing with fakes)
            receipt_parser: Optional custom receipt parser (for testing with fakes)

        Returns:
            Tuple of (successful_count, failed_count, total_token_usage, observability)
        """
        if not image_files:
            self.logger.warning("No images to process")
            return 0, 0, TokenUsage(), None

        total_images = len(image_files)
        self.logger.info(f"Processing {total_images} images with mode={self.config.mode.value}, "
                        f"max_workers={self.config.max_workers}")

        batch_start_time = time.time()

        # Track results
        results: List[ProcessingResult] = []
        ocr_breakdown: Dict[str, int] = {}

        # Use provided processors for tests, otherwise use worker-based processing
        if image_processor is not None and receipt_parser is not None:
            # Test mode: use provided fakes directly (sequential only)
            results = self._process_with_fakes(
                image_files, image_processor, receipt_parser, ocr_breakdown
            )
        elif self.config.max_workers > 1:
            # Parallel processing with ProcessPoolExecutor
            results = self._process_parallel(image_files, ocr_breakdown)
        else:
            # Sequential processing with fresh instances
            results = self._process_sequential(image_files, ocr_breakdown)

        # Aggregate results
        successful_processes = sum(1 for r in results if r.success)
        failed_processes = total_images - successful_processes

        total_token_usage = TokenUsage()
        for r in results:
            total_token_usage.add_usage(
                r.token_usage.input_tokens,
                r.token_usage.output_tokens
            )

        batch_time = (time.time() - batch_start_time) * 1000
        avg_time = batch_time / total_images if total_images > 0 else 0

        # Create observability summary
        observability = BatchObservability(
            total_images=total_images,
            successful=successful_processes,
            failed=failed_processes,
            total_time_ms=batch_time,
            avg_time_per_image_ms=avg_time,
            total_tokens=total_token_usage,
            ocr_breakdown=ocr_breakdown,
        )

        # Log detailed summary (debug only)
        self._log_batch_summary(results, observability)

        return successful_processes, failed_processes, total_token_usage, observability

    def _process_parallel(
        self,
        image_files: List[Path],
        ocr_breakdown: Dict[str, int],
    ) -> List[ProcessingResult]:
        """Process images in parallel using ProcessPoolExecutor"""
        results = []

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            for image_path in image_files:
                self._print_processing_header(image_path)
                future = executor.submit(
                    process_single_image,
                    str(image_path),
                    self.config.mode.value,
                    self.config.max_workers,
                    self.config.ocr_threads,
                )
                future_to_path[future] = image_path

            # Collect results as they complete
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Print formatted output
                    self._print_result(result)

                    # Track OCR method usage
                    ocr_breakdown[result.ocr_method] = ocr_breakdown.get(result.ocr_method, 0) + 1

                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    error_result = ProcessingResult(
                        image_path=str(image_path),
                        success=False,
                        parsed_data={},
                        retries=[],
                        validation_error=str(e),
                        token_usage=TokenUsage(),
                        processing_time_ms=0.0,
                        ocr_method="error",
                        ocr_duration_ms=0.0,
                        ocr_attempted_methods=[],
                        ocr_quality_score=0.0,
                    )
                    results.append(error_result)
                    ocr_breakdown["error"] = ocr_breakdown.get("error", 0) + 1

        return results

    def _process_sequential(
        self,
        image_files: List[Path],
        ocr_breakdown: Dict[str, int],
    ) -> List[ProcessingResult]:
        """Process images sequentially (single-threaded)"""
        results = []

        for image_path in image_files:
            self._print_processing_header(image_path)
            result = process_single_image(
                str(image_path),
                self.config.mode.value,
                self.config.max_workers,
                self.config.ocr_threads,
            )
            results.append(result)

            # Print formatted output
            self._print_result(result)

            # Track OCR method usage
            ocr_breakdown[result.ocr_method] = ocr_breakdown.get(result.ocr_method, 0) + 1

        return results

    def _print_processing_header(self, image_path: Path) -> None:
        """Print structured header before processing a receipt"""
        filename = image_path.name
        print(f"\n{'='*60}")
        print(f"PROCESSING: {filename}")
        print(f"{'='*60}")
        print("PIPELINE:")
        print("- OCR → Parsing → Validation")

    def _print_result(self, result: ProcessingResult) -> None:
        """Print formatted result for a single image"""
        token_data = {
            'input_tokens': result.token_usage.input_tokens,
            'output_tokens': result.token_usage.output_tokens,
            'total_tokens': result.token_usage.get_total_tokens(),
        }

        formatted = {
            'image_path': result.image_path,
            'success': result.success,
            'parsed_receipt': result.parsed_data,
            'retries': result.retries,
            'validation_error': result.validation_error,
            'token_usage': token_data,
            'processing_time_ms': round(result.processing_time_ms, 1),
            'ocr_method': result.ocr_method,
            'ocr_duration_ms': round(result.ocr_duration_ms, 1),
            'ocr_attempted_methods': result.ocr_attempted_methods,
            'ocr_quality_score': result.ocr_quality_score,
        }

        print(format_receipt_result(formatted))

    def _process_single_with_fakes(
        self,
        image_path: Path,
        image_processor: ImageProcessingInterface,
        receipt_parser: ReceiptParsingInterface,
        ocr_breakdown: Dict[str, int],
    ) -> ProcessingResult:
        """Process a single image using fake/test services."""
        start_time = time.time()

        try:
            ocr_text, ocr_duration = self._extract_text_with_timing(image_processor, image_path)
            parse_result = receipt_parser.parse_with_validation_driven_retry(ocr_text, str(image_path))
            parsed_data = _convert_parsed_data(parse_result)
            has_meaningful_data = (
                parse_result.valid and
                parsed_data and
                parsed_data.get('items') and
                len(parsed_data.get('items', [])) > 0 and
                parsed_data.get('total') is not None
            )
            processing_time = (time.time() - start_time) * 1000

            result = ProcessingResult(
                image_path=str(image_path),
                success=has_meaningful_data,
                parsed_data=parsed_data,
                retries=receipt_parser.get_current_retries(),
                validation_error=parse_result.error if parse_result.error else None,
                token_usage=parse_result.token_usage if parse_result.token_usage else TokenUsage(),
                processing_time_ms=processing_time,
                ocr_method='fake',
                ocr_duration_ms=ocr_duration,
                ocr_attempted_methods=['fake'],
                ocr_quality_score=0.0,
            )
            ocr_breakdown['fake'] = ocr_breakdown.get('fake', 0) + 1
            self._print_result(result)
            return result

        except Exception as e:
            self.logger.error(f"Failed to process {image_path}: {e}")
            ocr_breakdown["error"] = ocr_breakdown.get("error", 0) + 1
            return _build_error_result(str(image_path))

    def _extract_text_with_timing(
        self,
        image_processor: ImageProcessingInterface,
        image_path: Path
    ) -> tuple:
        """Extract OCR text and return text with duration in ms."""
        ocr_start = time.time()
        ocr_text = image_processor.extract_text(str(image_path))
        ocr_duration = (time.time() - ocr_start) * 1000
        return ocr_text, ocr_duration


    def _process_with_fakes(
        self,
        image_files: List[Path],
        image_processor: ImageProcessingInterface,
        receipt_parser: ReceiptParsingInterface,
        ocr_breakdown: Dict[str, int],
    ) -> List[ProcessingResult]:
        """Process images using fake services for testing (no worker processes)."""
        return [
            self._process_single_with_fakes(image_path, image_processor, receipt_parser, ocr_breakdown)
            for image_path in image_files
        ]

    def _log_batch_summary(
        self,
        results: List[ProcessingResult],
        obs: BatchObservability,
    ) -> None:
        """Log detailed batch summary at debug level to reduce output noise"""
        logger.debug("=" * 60)
        logger.debug("BATCH PROCESSING COMPLETE")
        logger.debug(f"Total: {obs.total_images} | Successful: {obs.successful} | Failed: {obs.failed}")
        logger.debug(f"Success Rate: {(obs.successful/obs.total_images*100):.1f}%")
        logger.debug(f"Total Time: {obs.total_time_ms:.1f}ms | Avg per image: {obs.avg_time_per_image_ms:.1f}ms")
        logger.debug(f"Tokens: {obs.total_tokens.get_total_tokens():,} total")
        logger.debug(f"OCR Methods: {obs.ocr_breakdown}")
        logger.debug("=" * 60)

    def validate_image_files(self, imgs_dir: Path) -> List[Path]:
        """Validate and get list of image files to process"""
        if not imgs_dir.exists():
            logger.error(f"Directory 'imgs' not found: {imgs_dir}")
            return []

        image_files = get_image_files(imgs_dir)

        if not image_files:
            logger.warning(f"No image files found in {imgs_dir}")
            return []

        logger.info(f"Found {len(image_files)} image files to process")
        return image_files

    def get_batch_summary(self, successful: int, failed: int, total: int) -> str:
        """Generate batch processing summary"""
        summary = f"""
==================================================
BATCH PROCESSING COMPLETE
Successful: {successful}
Failed: {failed}
Total: {total}
Success Rate: {(successful/total*100):.1f}%
==================================================
"""
        return summary

    def print_batch_summary(self, successful: int, failed: int, total: int):
        """Print batch processing summary (legacy - use print_batch_summary_clean instead)"""
        logger.info("=" * 50)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {total}")
        logger.info(f"Success Rate: {(successful/total*100):.1f}%")
        logger.info("=" * 50)

    def print_batch_summary_clean(self, observability: BatchObservability) -> None:
        """Print clean, readable batch summary"""
        total = observability.total_images
        successful = observability.successful
        failed = observability.failed
        success_rate = (successful / total * 100) if total > 0 else 0

        # Format times
        total_time_s = observability.total_time_ms / 1000
        avg_time_s = observability.avg_time_per_image_ms / 1000

        # Format OCR breakdown
        ocr_lines = []
        for method, count in sorted(observability.ocr_breakdown.items()):
            ocr_lines.append(f"  {method}: {count}")

        print()
        print("=" * 50)
        print("BATCH SUMMARY")
        print("=" * 50)
        print()
        print(f"Total: {total}")
        print(f"Success: {successful}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.0f}%")
        print()
        print("Performance:")
        print(f"  Total Time: {total_time_s:.1f}s")
        print(f"  Avg/Image: {avg_time_s:.1f}s")
        print()
        print("OCR Usage:")
        if ocr_lines:
            for line in ocr_lines:
                print(line)
        else:
            print("  N/A")
        print()
        print("Tokens:")
        print(f"  Total: {observability.total_tokens.get_total_tokens():,}")
        print()
        print("=" * 50)

    def print_processing_result(self, result: APIResponse, image_files: List[Path], index: int):
        """Print the result of processing a single image"""
        logger.info(f"Processing image: {image_files[index]}")

        if result.status == 'success':
            logger.info("SUCCESS")
            if result.data:
                logger.info(f"Parsed Receipt: {result.data}")
        else:
            logger.error(f"FAILED: {result.error}")
            if result.error:
                logger.error(f"Error: {result.error}")
