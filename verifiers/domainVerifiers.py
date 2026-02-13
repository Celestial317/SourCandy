from typing import List, Dict, Optional, Any, Set
from schema import SournessMap, TokenDiagnosis
from knowledge.graph import KnowledgeGraph
from knowledge.store import KnowledgeStore
from core.embedding import BaseEmbedder
import re


class DomainVerifier:
    '''
    Domain-specific verification for output validation.
    Checks generated content against domain-specific rules and constraints.
    Identifies hallucinations and factual inconsistencies within specialized domains.
    '''
    
    DEFAULT_ENTITY_SIMILARITY_THRESHOLD = 0.7
    DEFAULT_CLAIM_SIMILARITY_THRESHOLD = 0.6
    DEFAULT_SOURNESS_THRESHOLD = 0.7
    DEFAULT_CLUSTER_SIZE = 3
    
    def __init__(
        self,
        domain: str,
        knowledgeGraph: KnowledgeGraph,
        knowledgeStore: KnowledgeStore,
        embedder: BaseEmbedder,
        entitySimilarityThreshold: float = 0.7,
        claimSimilarityThreshold: float = 0.6,
        sournessThreshold: float = 0.7,
        clusterSize: int = 3
    ):
        '''
        Initialize domain verifier with knowledge sources.
        
        Args:
            domain: Domain identifier (e.g., medical, legal, technical)
            knowledgeGraph: Knowledge graph for entity verification
            knowledgeStore: Document store for factual grounding
            embedder: Embedding model for semantic matching
            entitySimilarityThreshold: Minimum similarity for entity verification (0.0-1.0)
            claimSimilarityThreshold: Minimum similarity for claim verification (0.0-1.0)
            sournessThreshold: Threshold for high sourness detection (0.0-1.0)
            clusterSize: Size of token cluster for sourness analysis (must be > 0)
            
        Raises:
            TypeError: If dependencies are of wrong type
            ValueError: If thresholds or parameters are out of valid range
        '''
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError('domain must be a non-empty string')
        if not isinstance(knowledgeGraph, KnowledgeGraph):
            raise TypeError('knowledgeGraph must be a KnowledgeGraph instance')
        if not isinstance(knowledgeStore, KnowledgeStore):
            raise TypeError('knowledgeStore must be a KnowledgeStore instance')
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError('embedder must be a BaseEmbedder instance')
        
        if not (0.0 <= entitySimilarityThreshold <= 1.0):
            raise ValueError(f'entitySimilarityThreshold must be between 0.0 and 1.0, got {entitySimilarityThreshold}')
        if not (0.0 <= claimSimilarityThreshold <= 1.0):
            raise ValueError(f'claimSimilarityThreshold must be between 0.0 and 1.0, got {claimSimilarityThreshold}')
        if not (0.0 <= sournessThreshold <= 1.0):
            raise ValueError(f'sournessThreshold must be between 0.0 and 1.0, got {sournessThreshold}')
        if not isinstance(clusterSize, int) or clusterSize <= 0:
            raise ValueError(f'clusterSize must be a positive integer, got {clusterSize}')
        
        self.domain = domain
        self.knowledgeGraph = knowledgeGraph
        self.knowledgeStore = knowledgeStore
        self.embedder = embedder
        self.entitySimilarityThreshold = entitySimilarityThreshold
        self.claimSimilarityThreshold = claimSimilarityThreshold
        self.sournessThreshold = sournessThreshold
        self.clusterSize = clusterSize
        self.domainKeywords: Set[str] = set()
        self.domainConstraints: List[Dict[str, Any]] = []
        
    def addDomainKeywords(self, keywords: List[str]):
        '''
        Register domain-specific keywords for enhanced verification.
        
        Args:
            keywords: List of domain-relevant terms
        '''
        self.domainKeywords.update(keyword.lower() for keyword in keywords)
        
    def addConstraint(self, constraintType: str, rule: Any):
        '''
        Add domain-specific constraint rule.
        
        Args:
            constraintType: Type of constraint (pattern, entity, numerical)
            rule: Constraint specification or validation function
        '''
        self.domainConstraints.append({
            'type': constraintType,
            'rule': rule
        })
        
    def verify(self, text: str, sournessMap: SournessMap) -> List[str]:
        '''
        Verify text against domain-specific constraints and knowledge.
        
        Args:
            text: Generated text to verify
            sournessMap: Token-level sourness diagnostics
            
        Returns:
            List of identified hallucinations or violations
        '''
        hallucinations = []
        
        entityHallucinations = self.verifyEntities(text)
        hallucinations.extend(entityHallucinations)
        
        factualHallucinations = self.verifyFactualClaims(text)
        hallucinations.extend(factualHallucinations)
        
        constraintViolations = self.verifyConstraints(text)
        hallucinations.extend(constraintViolations)
        
        sournessHallucinations = self.verifyHighSournessRegions(text, sournessMap)
        hallucinations.extend(sournessHallucinations)
        
        return list(set(hallucinations))
    
    def verifyEntities(self, text: str) -> List[str]:
        '''
        Check if mentioned entities exist in knowledge graph.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of hallucinated entity mentions
            
        Raises:
            RuntimeError: If entity verification fails
        '''
        hallucinations = []
        
        try:
            tokens = text.split()
            capitalizedPhrases = self.extractCapitalizedPhrases(text)
            
            for phrase in capitalizedPhrases:
                try:
                    phraseEmbedding = self.embedder.embedText(phrase)
                    matches = self.knowledgeGraph.findClosestEntity(phraseEmbedding, topK=1)
                    
                    if not matches or matches[0][1] < self.entitySimilarityThreshold:
                        hallucinations.append(f'Unverified entity: {phrase}')
                except Exception as e:
                    hallucinations.append(f'Failed to verify entity {phrase}: {str(e)}')
                    
            return hallucinations
        except Exception as e:
            raise RuntimeError(f'Entity verification failed: {str(e)}')
    
    def verifyFactualClaims(self, text: str) -> List[str]:
        '''
        Verify factual claims against knowledge store.
        
        Args:
            text: Text containing claims
            
        Returns:
            List of unsupported claims
            
        Raises:
            RuntimeError: If claim verification fails
        '''
        hallucinations = []
        
        try:
            sentences = self.extractSentences(text)
            
            for sentence in sentences:
                if self.containsFactualClaim(sentence):
                    try:
                        results = self.knowledgeStore.searchSimilar(sentence, topK=3)
                        
                        if not results or all(r['score'] < self.claimSimilarityThreshold for r in results):
                            hallucinations.append(f'Unsupported claim: {sentence[:100]}...')
                    except Exception as e:
                        hallucinations.append(f'Failed to verify claim: {str(e)}')
                        
            return hallucinations
        except Exception as e:
            raise RuntimeError(f'Factual claim verification failed: {str(e)}')
    
    def verifyConstraints(self, text: str) -> List[str]:
        '''
        Check text against registered domain constraints.
        
        Args:
            text: Text to validate
            
        Returns:
            List of constraint violations
        '''
        violations = []
        
        for constraint in self.domainConstraints:
            constraintType = constraint['type']
            rule = constraint['rule']
            
            if constraintType == 'pattern':
                if not re.search(rule, text):
                    violations.append(f'Missing required pattern: {rule}')
                    
            elif constraintType == 'forbidden':
                if re.search(rule, text):
                    violations.append(f'Contains forbidden pattern: {rule}')
                    
            elif constraintType == 'callable':
                if callable(rule):
                    try:
                        result = rule(text)
                        if not result:
                            violations.append('Failed custom validation rule')
                    except Exception as e:
                        violations.append(f'Constraint validation error: {str(e)}')
                        
        return violations
    
    def verifyHighSournessRegions(
        self,
        text: str,
        sournessMap: SournessMap
    ) -> List[str]:
        '''
        Identify high-sourness token clusters as potential hallucinations.
        
        Args:
            text: Generated text
            sournessMap: Token sourness diagnostics
            
        Returns:
            List of suspected hallucinated regions
            
        Raises:
            RuntimeError: If sourness region analysis fails
        '''
        hallucinations = []
        
        try:
            diagnostics = sournessMap.tokenDiagnostics
            
            for i in range(len(diagnostics) - self.clusterSize + 1):
                cluster = diagnostics[i:i+self.clusterSize]
                averageSourness = sum(d.sournessScore for d in cluster) / len(cluster)
                
                if averageSourness > self.sournessThreshold:
                    clusterText = ' '.join(d.token for d in cluster)
                    hallucinations.append(f'High sourness region: {clusterText}')
                    
            return hallucinations
        except Exception as e:
            raise RuntimeError(f'High sourness region analysis failed: {str(e)}')
    
    def extractCapitalizedPhrases(self, text: str) -> List[str]:
        '''
        Extract capitalized phrases likely to be named entities.
        
        Args:
            text: Input text
            
        Returns:
            List of capitalized phrase strings
            
        Raises:
            RuntimeError: If regex extraction fails
        '''
        try:
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            matches = re.findall(pattern, text)
            return [m for m in matches if len(m) > 3]
        except Exception as e:
            raise RuntimeError(f'Failed to extract capitalized phrases: {str(e)}')
    
    def extractSentences(self, text: str) -> List[str]:
        '''
        Split text into sentences for claim verification.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence strings
            
        Raises:
            RuntimeError: If sentence extraction fails
        '''
        try:
            sentencePattern = r'[^.!?]+[.!?]'
            sentences = re.findall(sentencePattern, text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        except Exception as e:
            raise RuntimeError(f'Failed to extract sentences: {str(e)}')
    
    def containsFactualClaim(self, sentence: str) -> bool:
        '''
        Determine if sentence contains a factual claim requiring verification.
        
        Args:
            sentence: Sentence text
            
        Returns:
            True if sentence appears to make factual claims
        '''
        factualIndicators = [
            r'\b(is|are|was|were)\b',
            r'\b(has|have|had)\b',
            r'\b\d+\b',
            r'\b(percent|percentage|million|billion)\b',
            r'\b(according to|research shows|studies indicate)\b'
        ]
        
        for pattern in factualIndicators:
            if re.search(pattern, sentence.lower()):
                return True
                
        return False


class MultiDomainVerifier:
    '''
    Aggregates multiple domain verifiers for comprehensive validation.
    Manages verification across different knowledge domains simultaneously.
    '''
    
    def __init__(self):
        '''
        Initialize multi-domain verifier with empty verifier registry.
        '''
        self.verifiers: Dict[str, DomainVerifier] = {}
        
    def addVerifier(self, domain: str, verifier: DomainVerifier):
        '''
        Register a domain-specific verifier.
        
        Args:
            domain: Domain identifier
            verifier: DomainVerifier instance for that domain
        '''
        self.verifiers[domain] = verifier
        
    def verifyAcrossDomains(
        self,
        text: str,
        sournessMap: SournessMap,
        activeDomains: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        '''
        Run verification across multiple domains.
        
        Args:
            text: Text to verify
            sournessMap: Token sourness diagnostics
            activeDomains: Optional list of domains to check. If None, checks all
            
        Returns:
            Dictionary mapping domain names to their hallucination lists
        '''
        domainsToCheck = activeDomains if activeDomains else list(self.verifiers.keys())
        
        results = {}
        for domain in domainsToCheck:
            if domain in self.verifiers:
                results[domain] = self.verifiers[domain].verify(text, sournessMap)
                
        return results
    
    def getAggregatedHallucinations(
        self,
        text: str,
        sournessMap: SournessMap
    ) -> List[str]:
        '''
        Get all hallucinations from all registered verifiers.
        
        Args:
            text: Text to verify
            sournessMap: Token sourness diagnostics
            
        Returns:
            Deduplicated list of all identified hallucinations
        '''
        allHallucinations = []
        
        for verifier in self.verifiers.values():
            hallucinations = verifier.verify(text, sournessMap)
            allHallucinations.extend(hallucinations)
            
        return list(set(allHallucinations))
