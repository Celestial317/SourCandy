'''
SourCandy: Token-level hallucination detection and correction framework.
Transform your LLM outputs from sour to sweet through granular diagnostic analysis.
'''

__version__ = '0.1.0'

from .guard import Guard
from .schema import (
    TokenDiagnosis,
    SournessMap,
    DiagnosticReport,
    SourCandyResponse
)
from .core.embedding import BaseEmbedder, SimpleEmbedder
from .core.sournessIndex import SournessCalculator
from .core.visualizer import SournessVisualizer, plotSournessHeatmap
from .knowledge.graph import KnowledgeGraph
from .knowledge.store import KnowledgeStore
from .correctors.fixes import SurgicalFixer
from .optimizers.promptRefiner import PromptRefiner
from .verifiers.domainVerifiers import DomainVerifier, MultiDomainVerifier

__all__ = [
    'Guard',
    'TokenDiagnosis',
    'SournessMap',
    'DiagnosticReport',
    'SourCandyResponse',
    'BaseEmbedder',
    'SimpleEmbedder',
    'SournessCalculator',
    'SournessVisualizer',
    'plotSournessHeatmap',
    'KnowledgeGraph',
    'KnowledgeStore',
    'SurgicalFixer',
    'PromptRefiner',
    'DomainVerifier',
    'MultiDomainVerifier',
]