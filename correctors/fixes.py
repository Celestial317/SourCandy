from typing import List, Optional, Any
import numpy as np
from schema import SournessMap, TokenDiagnosis
from knowledge.graph import KnowledgeGraph
from knowledge.store import KnowledgeStore
from core.embedding import BaseEmbedder


BaseLLM = Any


class SurgicalFixer:
    '''
    Performs surgical token-level correction on sour outputs.
    Identifies tokens exceeding sourness threshold and replaces them
    with grounded alternatives from the knowledge graph while maintaining
    semantic flow using LLM-based infilling.
    '''
    
    DEFAULT_SOURNESS_THRESHOLD = 0.6
    DEFAULT_CONTEXT_WINDOW = 3
    DEFAULT_TOP_K_CANDIDATES = 3
    INFILL_MODIFICATION_THRESHOLD = 2
    
    def __init__(
        self,
        baseLlm: BaseLLM,
        knowledgeGraph: KnowledgeGraph,
        knowledgeStore: KnowledgeStore,
        embedder: BaseEmbedder,
        sournessThreshold: float = 0.6,
        contextWindow: int = 3,
        topKCandidates: int = 3
    ):
        '''
        Initialize surgical fixer with required dependencies.
        
        Args:
            baseLlm: Language model for semantic infilling
            knowledgeGraph: Knowledge graph for entity retrieval
            knowledgeStore: Vector store for document grounding
            embedder: Embedding model for similarity matching
            sournessThreshold: Threshold above which tokens are considered sour (0.0-1.0)
            contextWindow: Number of context tokens before/after for replacement (must be > 0)
            topKCandidates: Number of candidate replacements to consider (must be > 0)
            
        Raises:
            TypeError: If dependencies are of wrong type
            ValueError: If thresholds or parameters are out of valid range
        '''
        if baseLlm is None:
            raise ValueError('baseLlm cannot be None')
        if not isinstance(knowledgeGraph, KnowledgeGraph):
            raise TypeError('knowledgeGraph must be a KnowledgeGraph instance')
        if not isinstance(knowledgeStore, KnowledgeStore):
            raise TypeError('knowledgeStore must be a KnowledgeStore instance')
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError('embedder must be a BaseEmbedder instance')
        
        if not (0.0 <= sournessThreshold <= 1.0):
            raise ValueError(f'sournessThreshold must be between 0.0 and 1.0, got {sournessThreshold}')
        if not isinstance(contextWindow, int) or contextWindow <= 0:
            raise ValueError(f'contextWindow must be a positive integer, got {contextWindow}')
        if not isinstance(topKCandidates, int) or topKCandidates <= 0:
            raise ValueError(f'topKCandidates must be a positive integer, got {topKCandidates}')
        
        self.baseLlm = baseLlm
        self.knowledgeGraph = knowledgeGraph
        self.knowledgeStore = knowledgeStore
        self.embedder = embedder
        self.sournessThreshold = sournessThreshold
        self.contextWindow = contextWindow
        self.topKCandidates = topKCandidates
        
    def fixSourTokens(
        self,
        originalText: str,
        sournessMap: SournessMap,
        promptContext: str
    ) -> str:
        '''
        Perform surgical correction on sour tokens in text.
        Iterates through sourness map and replaces high-sourness tokens
        with knowledge-grounded alternatives.
        
        Args:
            originalText: The raw generated text
            sournessMap: Token-level diagnostics from SournessCalculator
            promptContext: Original prompt for context preservation
            
        Returns:
            Corrected text with sour tokens replaced
            
        Raises:
            TypeError: If inputs are of wrong type
            RuntimeError: If correction process fails
        '''
        if not isinstance(originalText, str):
            raise TypeError(f'originalText must be a string, got {type(originalText).__name__}')
        if not isinstance(sournessMap, SournessMap):
            raise TypeError('sournessMap must be a SournessMap instance')
        if not isinstance(promptContext, str):
            raise TypeError(f'promptContext must be a string, got {type(promptContext).__name__}')
        
        try:
            tokens = originalText.split()
            
            if len(tokens) != len(sournessMap.tokenDiagnostics):
                return originalText
                
            fixedTokens = []
            modificationsNeeded = []
            
            for idx, diagnosis in enumerate(sournessMap.tokenDiagnostics):
                if diagnosis.sournessScore > self.sournessThreshold:
                    modificationsNeeded.append(idx)
                fixedTokens.append(diagnosis.token)
                
            if not modificationsNeeded:
                return originalText
                
            for idx in modificationsNeeded:
                diagnosis = sournessMap.tokenDiagnostics[idx]
                replacement = self.findReplacementToken(
                    sourToken=diagnosis.token,
                    groundingIds=diagnosis.groundingNodeIds,
                    contextBefore=fixedTokens[max(0, idx-self.contextWindow):idx],
                    contextAfter=fixedTokens[idx+1:min(len(fixedTokens), idx+self.contextWindow+1)]
                )
                
                if replacement:
                    fixedTokens[idx] = replacement
                    
            preliminaryText = ' '.join(fixedTokens)
            
            if len(modificationsNeeded) > self.INFILL_MODIFICATION_THRESHOLD:
                finalText = self.llmInfill(
                    text=preliminaryText,
                    promptContext=promptContext,
                    modifiedIndices=modificationsNeeded
                )
            else:
                finalText = preliminaryText
                
            return finalText
        except Exception as e:
            raise RuntimeError(f'Failed to fix sour tokens: {str(e)}')
    
    def findReplacementToken(
        self,
        sourToken: str,
        groundingIds: List[str],
        contextBefore: List[str],
        contextAfter: List[str]
    ) -> Optional[str]:
        '''
        Find knowledge-grounded replacement for a sour token.
        Queries knowledge graph for entities and selects contextually appropriate term.
        
        Args:
            sourToken: The token to replace
            groundingIds: Knowledge graph entity IDs associated with token
            contextBefore: Preceding tokens for context
            contextAfter: Following tokens for context
            
        Returns:
            Replacement token or None if no suitable candidate found
            
        Raises:
            RuntimeError: If replacement search fails
        '''
        try:
            if not groundingIds:
                contextQuery = ' '.join(contextBefore + [sourToken] + contextAfter)
                tokenEmbedding = self.embedder.embedText(contextQuery)
                candidates = self.knowledgeGraph.findClosestEntity(tokenEmbedding, topK=self.topKCandidates)
                
                if candidates:
                    return self.knowledgeGraph.entityLabels.get(candidates[0][0], sourToken)
                return None
                
            primaryEntity = groundingIds[0]
            
            if primaryEntity in self.knowledgeGraph.entityLabels:
                label = self.knowledgeGraph.entityLabels[primaryEntity]
                if label != sourToken:
                    return label
                    
            subgraph = self.knowledgeGraph.getRelatedSubGraph([primaryEntity], maxDepth=1)
            
            if len(subgraph) > 1:
                contextEmbedding = self.embedder.embedText(' '.join(contextBefore + contextAfter))
                
                bestScore = -1.0
                bestCandidate = None
                
                for entityId, embedding in subgraph.items():
                    if entityId == primaryEntity:
                        continue
                    
                similarity = np.dot(contextEmbedding, embedding) / (
                    np.linalg.norm(contextEmbedding) * np.linalg.norm(embedding) + 1e-8
                )
                
                if similarity > bestScore:
                    bestScore = similarity
                    bestCandidate = self.knowledgeGraph.entityLabels.get(entityId, None)
                    
            if bestCandidate:
                return bestCandidate
                
            return None
        except Exception as e:
            raise RuntimeError(f'Failed to find replacement token: {str(e)}')
    
    def llmInfill(
        self,
        text: str,
        promptContext: str,
        modifiedIndices: List[int]
    ) -> str:
        '''
        Use LLM to smooth and improve fluency after token replacements.
        Maintains semantic meaning while fixing grammatical inconsistencies.
        
        Args:
            text: Text with replaced tokens
            promptContext: Original prompt for grounding
            modifiedIndices: Indices of modified tokens
            
        Returns:
            Semantically smoothed text (returns original on failure)
        '''
        infillPrompt = f'''Given the original request: {promptContext}

The following text has been corrected for factual accuracy but may have awkward phrasing:
{text}

Please rewrite this text to maintain the factual corrections while improving fluency and naturalness. Keep all key facts and entities intact.

Improved version:'''
        
        try:
            if hasattr(self.baseLlm, 'invoke'):
                response = self.baseLlm.invoke(infillPrompt)
                if hasattr(response, 'content'):
                    return response.content.strip()
                return str(response).strip()
            elif callable(self.baseLlm):
                return str(self.baseLlm(infillPrompt)).strip()
            else:
                return text
        except Exception as e:
            return text
    
    def batchFix(
        self,
        texts: List[str],
        sournessMaps: List[SournessMap],
        promptContexts: List[str]
    ) -> List[str]:
        '''
        Apply surgical fixing to multiple texts efficiently.
        
        Args:
            texts: List of generated texts
            sournessMaps: Corresponding sourness maps
            promptContexts: Original prompts for each text
            
        Returns:
            List of corrected texts
        '''
        if len(texts) != len(sournessMaps) or len(texts) != len(promptContexts):
            raise ValueError('Input lists must have equal length')
            
        return [
            self.fixSourTokens(text, sMap, prompt)
            for text, sMap, prompt in zip(texts, sournessMaps, promptContexts)
        ]
