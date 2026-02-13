from typing import List, Dict, Optional, Any
import numpy as np
from schema import TokenDiagnosis, SournessMap
from knowledge.graph import KnowledgeGraph
from core.embedding import BaseEmbedder


class SournessCalculator:
    '''
    Token-level sourness scoring using Modified Cross Attention mechanism.
    Implements the research formula: S_t = 1 - tanh(w_1*G_t + w_2*P_t + w_3*R_t)
    where G_t is knowledge grounding, P_t is prompt correspondence, R_t is relational connectivity.
    '''
    
    DEFAULT_W1 = 0.5
    DEFAULT_W2 = 0.3
    DEFAULT_W3 = 0.2
    DEFAULT_EPSILON = 0.15
    DEFAULT_THRESHOLD = 0.6
    
    def __init__(
        self, 
        knowledgeGraph: KnowledgeGraph,
        embedder: BaseEmbedder,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        epsilon: float = 0.15
    ):
        '''
        Initialize sourness calculator with weights and dependencies.
        
        Args:
            knowledgeGraph: Knowledge graph for entity grounding
            embedder: Embedding model for token vectorization
            w1: Weight for knowledge grounding component G_t (0.0 to 1.0)
            w2: Weight for prompt correspondence component P_t (0.0 to 1.0)
            w3: Weight for relational connectivity component R_t (0.0 to 1.0)
            epsilon: Penalty applied when relational connectivity fails (0.0 to 1.0)
            
        Raises:
            TypeError: If dependencies are of wrong type
            ValueError: If weights or epsilon are out of valid range
        '''
        if not isinstance(knowledgeGraph, KnowledgeGraph):
            raise TypeError('knowledgeGraph must be a KnowledgeGraph instance')
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError('embedder must be a BaseEmbedder instance')
        
        if not (0.0 <= w1 <= 1.0):
            raise ValueError(f'w1 must be between 0.0 and 1.0, got {w1}')
        if not (0.0 <= w2 <= 1.0):
            raise ValueError(f'w2 must be between 0.0 and 1.0, got {w2}')
        if not (0.0 <= w3 <= 1.0):
            raise ValueError(f'w3 must be between 0.0 and 1.0, got {w3}')
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f'epsilon must be between 0.0 and 1.0, got {epsilon}')
        
        weightSum = w1 + w2 + w3
        if not (0.8 <= weightSum <= 1.2):
            raise ValueError(f'Sum of weights (w1+w2+w3={weightSum:.2f}) should be close to 1.0')
        
        self.knowledgeGraph = knowledgeGraph
        self.embedder = embedder
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.epsilon = epsilon
        self.tokenEmbeddingCache: Dict[str, np.ndarray] = {}
        
    def calcG(self, tokenEmbedding: np.ndarray, kgMatrix: np.ndarray) -> float:
        '''
        Calculate G_t: Knowledge Grounding score via softmax alignment.
        Measures max attention between token query and knowledge base keys.
        
        Args:
            tokenEmbedding: Query vector Q_t for the token
            kgMatrix: Knowledge base key matrix K_G of shape (n_entities, d)
            
        Returns:
            Maximum softmax alignment score (0 to 1)
            
        Raises:
            ValueError: If input arrays have invalid shapes
        '''
        if not isinstance(tokenEmbedding, np.ndarray):
            raise TypeError('tokenEmbedding must be a numpy array')
        if not isinstance(kgMatrix, np.ndarray):
            raise TypeError('kgMatrix must be a numpy array')
        
        if kgMatrix.shape[0] == 0:
            return 0.0
            
        try:
            similarities = []
            tokenNorm = np.linalg.norm(tokenEmbedding)
            
            if tokenNorm == 0:
                return 0.0
                
            normalizedToken = tokenEmbedding / tokenNorm
            
            for kgVector in kgMatrix:
                kgNorm = np.linalg.norm(kgVector)
                if kgNorm > 0:
                    similarity = np.dot(normalizedToken, kgVector / kgNorm)
                    similarities.append(similarity)
                else:
                    similarities.append(0.0)
                    
            if not similarities:
                return 0.0
                
            similarities = np.array(similarities)
            expScores = np.exp(similarities - np.max(similarities))
            softmaxScores = expScores / np.sum(expScores)
            
            return float(np.max(softmaxScores))
        except Exception as e:
            raise RuntimeError(f'Failed to calculate G_t score: {str(e)}')
    
    def calcP(self, tokenEmbedding: np.ndarray, promptEmbedding: np.ndarray) -> float:
        '''
        Calculate P_t: Prompt Correspondence score.
        Measures alignment between token and input prompt embedding.
        
        Args:
            tokenEmbedding: Query vector Q_t for the token
            promptEmbedding: Input prompt embedding K_P
            
        Returns:
            Cosine similarity between token and prompt (0 to 1)
        '''
        tokenNorm = np.linalg.norm(tokenEmbedding)
        promptNorm = np.linalg.norm(promptEmbedding)
        
        if tokenNorm == 0 or promptNorm == 0:
            return 0.0
            
        similarity = np.dot(tokenEmbedding / tokenNorm, promptEmbedding / promptNorm)
        
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))
    
    def calcR(
        self, 
        currentTokenGrounding: Optional[str], 
        previousTokenGrounding: Optional[str]
    ) -> float:
        '''
        Calculate R_t: Relational Connectivity score via neurosymbolic check.
        R_t = 1 if knowledge graph path exists between consecutive token groundings.
        R_t = epsilon (penalty) if no path exists.
        
        Args:
            currentTokenGrounding: Entity ID grounding current token
            previousTokenGrounding: Entity ID grounding previous token
            
        Returns:
            1.0 if connected, epsilon penalty otherwise
        '''
        if currentTokenGrounding is None or previousTokenGrounding is None:
            return self.epsilon
            
        if currentTokenGrounding == previousTokenGrounding:
            return 1.0
            
        hasConnection = self.knowledgeGraph.hasPath(
            previousTokenGrounding, 
            currentTokenGrounding,
            maxHops=3
        )
        
        return 1.0 if hasConnection else self.epsilon
    
    def calculateTokenSourness(
        self,
        token: str,
        tokenEmbedding: np.ndarray,
        promptEmbedding: np.ndarray,
        kgMatrix: np.ndarray,
        groundingEntityId: Optional[str],
        previousGroundingEntityId: Optional[str]
    ) -> float:
        '''
        Calculate sourness score S_t for a single token using the research formula.
        S_t = 1 - tanh(w_1*G_t + w_2*P_t + w_3*R_t)
        
        Args:
            token: The token string
            tokenEmbedding: Vector embedding of the token
            promptEmbedding: Embedding of the input prompt
            kgMatrix: Knowledge graph embedding matrix
            groundingEntityId: Entity ID that grounds this token
            previousGroundingEntityId: Entity ID that grounded previous token
            
        Returns:
            Sourness score between 0 (sweet) and 1 (sour)
        '''
        gScore = self.calcG(tokenEmbedding, kgMatrix)
        pScore = self.calcP(tokenEmbedding, promptEmbedding)
        rScore = self.calcR(groundingEntityId, previousGroundingEntityId)
        
        combinedScore = self.w1 * gScore + self.w2 * pScore + self.w3 * rScore
        sournessScore = 1.0 - np.tanh(combinedScore)
        
        return float(max(0.0, min(1.0, sournessScore)))
    
    def generateSournessMap(
        self,
        outputText: str,
        promptText: str
    ) -> SournessMap:
        '''
        Generate complete sourness map for all tokens in output text.
        Returns TokenDiagnosis objects for each token with grounding information.
        
        Args:
            outputText: Generated text to analyze
            promptText: Original input prompt
            
        Returns:
            SournessMap containing token diagnostics and aggregate score
            
        Raises:
            TypeError: If inputs are not strings
            RuntimeError: If sourness map generation fails
        '''
        if not isinstance(outputText, str):
            raise TypeError(f'outputText must be a string, got {type(outputText).__name__}')
        if not isinstance(promptText, str):
            raise TypeError(f'promptText must be a string, got {type(promptText).__name__}')
        
        try:
            tokens = outputText.split()
            
            if not tokens:
                return SournessMap(
                    tokenDiagnostics=[],
                    aggregateSournessScore=0.0
                )
                
            promptEmbedding = self.embedder.embedText(promptText)
            kgMatrix = self.knowledgeGraph.getEmbeddingMatrix()
            
            tokenDiagnostics = []
            previousGrounding = None
            totalSourness = 0.0
            
            for token in tokens:
                tokenEmbedding = self.embedder.embedText(token)
                
                closestEntities = self.knowledgeGraph.findClosestEntity(tokenEmbedding, topK=1)
                
                if closestEntities:
                    currentGrounding = closestEntities[0][0]
                    groundingNodeIds = [currentGrounding]
                else:
                    currentGrounding = None
                    groundingNodeIds = []
                    
                sournessScore = self.calculateTokenSourness(
                    token=token,
                    tokenEmbedding=tokenEmbedding,
                    promptEmbedding=promptEmbedding,
                    kgMatrix=kgMatrix,
                    groundingEntityId=currentGrounding,
                    previousGroundingEntityId=previousGrounding
                )
                
                tokenDiagnostics.append(TokenDiagnosis(
                    token=token,
                    sournessScore=sournessScore,
                    groundingNodeIds=groundingNodeIds
                ))
                
                totalSourness += sournessScore
                previousGrounding = currentGrounding
                
            aggregateScore = totalSourness / len(tokens) if tokens else 0.0
            
            return SournessMap(
                tokenDiagnostics=tokenDiagnostics,
                aggregateSournessScore=aggregateScore
            )
        except Exception as e:
            raise RuntimeError(f'Failed to generate sourness map: {str(e)}')
