import warnings
import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Also add current directory for relative imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:  # pragma: no cover
    NotOpenSSLWarning = Warning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)
