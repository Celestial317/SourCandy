"""
SourCandy: Make your models taste sweet.
"""

__version__ = "0.1.0"

from .guard import Guard
from .interfaces import (
    BaseVerifier,
    BaseCorrector,
    VerificationResult,
    CorrectionResult,
    SourCandyResponse
)

# This makes the main classes available when a user imports the package:
# e.g., from sourcandy import Guard
__all__ = [
    "Guard",
    "BaseVerifier",
    "BaseCorrector",
    "VerificationResult",
    "CorrectionResult",
    "SourCandyResponse",
]