from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TokenDiagnosis(BaseModel):
    token: str
    sournessScore: float
    groundingNodeIds: List[str]

class SournessMap(BaseModel):
    tokenDiagnostics: List[TokenDiagnosis]
    aggregateSournessScore: float

class DiagnosticReport(BaseModel):
    rawOutput: str
    sournessMap: SournessMap
    identifiedHallucinations: List[str]
    optimizationSuggestion: Optional[str]

class SourCandyResponse(BaseModel):
    finalContent: str
    isSweet: bool
    diagnosticReport: DiagnosticReport