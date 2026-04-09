"""Clean output formatter for receipt processing results"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from decimal import Decimal


def _get_field(obj: Union[dict, Any], field: str, default=None):
    """Get field value from dict or object attribute"""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def format_receipt_result(result: Dict[str, Any]) -> str:
    """Format receipt processing result into clean, human-readable output"""

    # Extract result components
    image_path = result.get('image_path', 'Unknown')
    success = result.get('success', False)
    parsed_receipt = result.get('parsed_receipt', {})
    retries = result.get('retries', [])
    validation_error = result.get('validation_error')
    token_usage = result.get('token_usage', {})

    # Get filename from path
    filename = Path(image_path).name

    # Build formatted output
    output = []
    output.append("=" * 30)
    output.append(f"RECEIPT: {filename}")
    output.append("=" * 30)
    output.append("")

    # Status
    status_icon = "✅ SUCCESS" if success else "❌ FAILED"
    output.append(f"STATUS: {status_icon}")

    # Retries
    if retries:
        retry_str = ", ".join(retries)
    else:
        retry_str = "NONE"
    output.append(f"RETRIES: {retry_str}")

    # OCR decision trace
    ocr_method = result.get('ocr_method', 'unknown')
    ocr_attempted = result.get('ocr_attempted_methods', [])
    ocr_quality = result.get('ocr_quality_score', 0.0)

    if ocr_attempted:
        attempted_str = " → ".join(ocr_attempted)
    else:
        attempted_str = ocr_method

    output.append("")
    output.append("OCR:")
    output.append(f"  Attempted: {attempted_str}")
    output.append(f"  Selected: {ocr_method}")
    output.append(f"  Quality Score: {ocr_quality:.2f}")
    output.append("")

    # Receipt details - display parsed data regardless of validation status
    if parsed_receipt:
        total = parsed_receipt.get('total')
        items = parsed_receipt.get('items', [])

        # Defensive logging: if parsed exists but items are empty
        if not items:
            import logging
            logging.getLogger(__name__).warning(f"Parsed receipt exists but items list is empty for {image_path}")

        # Total
        if total is not None:
            output.append(f"TOTAL: ${total}")
        else:
            output.append("TOTAL: N/A")
        output.append("")

        # Items
        output.append(f"ITEMS ({len(items)}):")
        for item in items:
            description = _get_field(item, 'description', 'Unknown item')
            price = _get_field(item, 'price')
            if price is not None:
                # Align prices to the right
                price_str = f"${price:.2f}"
                output.append(f"- {description} .... {price_str}")
            else:
                output.append(f"- {description} .... N/A")

        # Item sum and mismatch
        if items:
            # Calculate sum using Decimal for consistency
            item_sum = Decimal('0')
            for item in items:
                price = _get_field(item, 'price')
                if price is not None:
                    try:
                        item_sum += Decimal(str(price))
                    except (ValueError, TypeError):
                        continue
            output.append("")
            output.append(f"ITEM SUM: ${item_sum:.2f}")

            if total is not None:
                # Convert total to Decimal for consistent arithmetic
                total_decimal = total if isinstance(total, Decimal) else Decimal(str(total))
                mismatch = abs(total_decimal - item_sum)
                if mismatch > Decimal('0.01'):  # Only show if significant mismatch
                    output.append(f"MISMATCH: ${mismatch:.2f}")
    else:
        output.append("TOTAL: N/A")
        output.append("")
        output.append("ITEMS (0):")

    # Token usage
    if token_usage:
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        total_tokens = token_usage.get('total_tokens', input_tokens + output_tokens)

        output.append("")
        output.append("TOKENS:")
        output.append(f"- Input: {input_tokens}")
        output.append(f"- Output: {output_tokens}")
        output.append(f"- Total: {total_tokens}")

    # Validation summary
    if validation_error:
        output.append("")
        output.append("VALIDATION:")

        # Parse mismatch information from error message
        import re
        mismatch_pattern = r'Total ([\d.,]+) does not match sum of items ([\d.,]+)'
        match = re.search(mismatch_pattern, str(validation_error))

        if match:
            total = match.group(1)
            items_sum = match.group(2)
            try:
                diff = abs(Decimal(total.replace(',', '.')) - Decimal(items_sum.replace(',', '.')))
                output.append(f"  Status: FAILED")
                output.append(f"  Reason: Total mismatch (${total} vs ${items_sum})")
                output.append(f"  Difference: ${diff:.2f}")
            except (ValueError, TypeError):
                output.append(f"  Status: FAILED")
                output.append(f"  Reason: Total mismatch")
        else:
            output.append(f"  Status: FAILED")
            # Extract clean reason from error message
            reason = str(validation_error).replace("Validation failed: ", "").replace("Self-correction validation failed: ", "").replace("RAG validation failed: ", "").replace("OCR fallback validation failed: ", "").replace("Response content validation failed: ", "")
            if len(reason) > 80:
                reason = reason[:77] + "..."
            output.append(f"  Reason: {reason}")

    output.append("")
    return "\n".join(output)


def calculate_item_sum(items: List[Any]) -> Decimal:
    """Calculate sum of item prices"""
    total = Decimal('0')
    for item in items:
        price = _get_field(item, 'price')
        if price is not None:
            try:
                total += Decimal(str(price))
            except (ValueError, TypeError):
                continue
    return total


def format_items_list(items: List[Any]) -> List[str]:
    """Format items list with aligned prices"""
    formatted_items = []

    for item in items:
        description = _get_field(item, 'description', 'Unknown item')
        price = _get_field(item, 'price')

        if price is not None:
            try:
                price_str = f"${float(price):.2f}"
                # Align description and price
                formatted_line = f"- {description} .... {price_str}"
            except (ValueError, TypeError):
                formatted_line = f"- {description} .... N/A"
        else:
            formatted_line = f"- {description} .... N/A"

        formatted_items.append(formatted_line)

    return formatted_items
