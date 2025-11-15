from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ---
# Data Structures
# We use Pydantic for robust data validation and serialization.
# ---

class VerificationResult(BaseModel):
    """Data structure for the output of a Verifier."""
    is_sour: bool = Field(..., description="True if a hallucination or error was detected.")
    sourness_type: str = Field(..., description="Type of sourness, e.g., 'factual', 'math', 'code_bug'.")
    reason: str = Field(..., description="A human-readable explanation of the detected issue.")
    confidence_score: float = Field(..., description="The verifier's confidence in its finding (0.0 to 1.0).")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra data, e.g., sources checked.")

    class Config:
        anystr_strip_whitespace = True


class CorrectionResult(BaseModel):
    """Data structure for the output of a Corrector."""
    is_fixed: bool = Field(..., description="True if the sourness was successfully corrected.")
    corrected_text: str = Field(..., description="The new, 'sweetened' text.")
    attempts: int = Field(..., description="Number of attempts made to fix the issue.")
    intermediate_steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="A log of the correction process.")

    class Config:
        anystr_strip_whitespace = True


class SourCandyResponse(BaseModel):
    """The final object returned by the Guard to the user."""
    content: str = Field(..., description="The final, 'sweetened' output after all checks.")
    sour_report: Dict[str, Any] = Field(..., description="A JSON-serializable log of all verification and correction actions.")
    raw_output: str = Field(..., description="The initial raw output from the base LLM.")
    is_final_sour: bool = Field(..., description="True if the final content is still sour (correction failed).")

    class Config:
        anystr_strip_whitespace = True

# ---
# Abstract Base Classes (ABCs)
# These are the "contracts" all our modules must follow.
# ---

class BaseVerifier(ABC):
    """Abstract Base Class for all Verifiers."""
    
    @abstractmethod
    def verify(self, text_input: str, context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """
        Verifies a piece of text for a specific type of sourness.

        Args:
            text_input: The text output from the LLM to verify.
            context: Optional dictionary containing extra info like the 
                     original user prompt, test files, or style guides.

        Returns:
            A VerificationResult object.
        """
        pass

class BaseCorrector(ABC):
    """Abstract Base Class for all Correctors."""
    
    @abstractabstractmethod
    def correct(self, text_input: str, result: VerificationResult) -> CorrectionResult:
        """
        Attempts to fix the "sourness" reported in the VerificationResult.

        Args:
            text_input: The sour text that needs fixing.
            result: The VerificationResult object that flagged the issue.

        Returns:
            A CorrectionResult object.
        """
        pass