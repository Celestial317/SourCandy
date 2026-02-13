from typing import Any, Dict, List, Optional
from schema import SourCandyResponse, DiagnosticReport, SournessMap
from core.embedding import BaseEmbedder, SimpleEmbedder
from core.sournessIndex import SournessCalculator
from knowledge.graph import KnowledgeGraph
from knowledge.store import KnowledgeStore
from correctors.fixes import SurgicalFixer
from optimizers.promptRefiner import PromptRefiner
from verifiers.domainVerifiers import DomainVerifier, MultiDomainVerifier


BaseLLM = Any


class Guard:
    '''
    Main orchestrator for the SourCandy framework.
    Implements end-to-end token-level diagnostic and surgical correction pipeline.
    Transforms raw LLM outputs into verified, grounded responses through:
    1. Input prompt analysis and knowledge retrieval
    2. Base LLM invocation
    3. Token-level sourness calculation
    4. Surgical correction of hallucinations
    5. Domain verification and optimization
    '''
    
    DEFAULT_SOURNESS_THRESHOLD = 0.6
    DEFAULT_W1 = 0.5
    DEFAULT_W2 = 0.3
    DEFAULT_W3 = 0.2
    
    def __init__(
        self,
        baseLlm: BaseLLM,
        knowledgeGraph: KnowledgeGraph,
        embedder: Optional[BaseEmbedder] = None,
        sournessThreshold: float = 0.6,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2
    ):
        '''
        Initialize SourCandy Guard with all pipeline components.
        
        Args:
            baseLlm: Base language model to wrap and monitor
            knowledgeGraph: Populated knowledge graph for grounding
            embedder: Optional custom embedder (defaults to SimpleEmbedder)
            sournessThreshold: Threshold above which tokens are corrected (0.0-1.0)
            w1: Weight for knowledge grounding component (0.0-1.0)
            w2: Weight for prompt correspondence component (0.0-1.0)
            w3: Weight for relational connectivity component (0.0-1.0)
            
        Raises:
            TypeError: If dependencies are of wrong type
            ValueError: If thresholds or weights are out of valid range
        '''
        if baseLlm is None:
            raise ValueError('baseLlm cannot be None')
        if not isinstance(knowledgeGraph, KnowledgeGraph):
            raise TypeError('knowledgeGraph must be a KnowledgeGraph instance')
        if embedder is not None and not isinstance(embedder, BaseEmbedder):
            raise TypeError('embedder must be a BaseEmbedder instance or None')
        
        if not (0.0 <= sournessThreshold <= 1.0):
            raise ValueError(f'sournessThreshold must be between 0.0 and 1.0, got {sournessThreshold}')
        if not (0.0 <= w1 <= 1.0):
            raise ValueError(f'w1 must be between 0.0 and 1.0, got {w1}')
        if not (0.0 <= w2 <= 1.0):
            raise ValueError(f'w2 must be between 0.0 and 1.0, got {w2}')
        if not (0.0 <= w3 <= 1.0):
            raise ValueError(f'w3 must be between 0.0 and 1.0, got {w3}')
        
        self.baseLlm = baseLlm
        self.knowledgeGraph = knowledgeGraph
        self.embedder = embedder if embedder else SimpleEmbedder()
        self.sournessThreshold = sournessThreshold
        
        self.knowledgeStore = KnowledgeStore(self.embedder)
        
        self.sournessCalculator = SournessCalculator(
            knowledgeGraph=self.knowledgeGraph,
            embedder=self.embedder,
            w1=w1,
            w2=w2,
            w3=w3
        )
        
        self.surgicalFixer = SurgicalFixer(
            baseLlm=self.baseLlm,
            knowledgeGraph=self.knowledgeGraph,
            knowledgeStore=self.knowledgeStore,
            embedder=self.embedder,
            sournessThreshold=self.sournessThreshold
        )
        
        self.promptRefiner = PromptRefiner(baseLlm=self.baseLlm)
        
        self.verifier = MultiDomainVerifier()
        
        self.runHistory: List[Dict[str, Any]] = []
        
    def addKnowledgeDocuments(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        '''
        Add documents to knowledge store for factual grounding.
        
        Args:
            documents: List of source documents
            metadata: Optional metadata for each document
            
        Raises:
            TypeError: If documents is not a list
            ValueError: If documents is empty
            RuntimeError: If adding documents fails
        '''
        if not isinstance(documents, list):
            raise TypeError(f'documents must be a list, got {type(documents).__name__}')
        if not documents:
            raise ValueError('documents list cannot be empty')
        
        try:
            self.knowledgeStore.addDocuments(documents, metadata)
        except Exception as e:
            raise RuntimeError(f'Failed to add knowledge documents: {str(e)}')
        
    def addDomainVerifier(self, domain: str, verifier: DomainVerifier):
        '''
        Register domain-specific verifier for specialized validation.
        
        Args:
            domain: Domain identifier
            verifier: Configured DomainVerifier instance
            
        Raises:
            TypeError: If verifier is not a DomainVerifier instance
            ValueError: If domain is empty
        '''
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError('domain must be a non-empty string')
        if not isinstance(verifier, DomainVerifier):
            raise TypeError('verifier must be a DomainVerifier instance')
        
        self.verifier.addVerifier(domain, verifier)
        
    def invoke(
        self,
        prompt: str,
        enableCorrection: bool = True,
        enableVerification: bool = True,
        returnDiagnostics: bool = True
    ) -> SourCandyResponse:
        '''
        Execute complete SourCandy pipeline on input prompt.
        
        Pipeline stages:
        1. Call base LLM with prompt
        2. Generate token-level sourness map
        3. Identify hallucinations via domain verification
        4. Apply surgical corrections to sour tokens
        5. Generate diagnostic report and optimization suggestions
        
        Args:
            prompt: User input prompt
            enableCorrection: Whether to apply surgical fixing
            enableVerification: Whether to run domain verification
            returnDiagnostics: Whether to include full diagnostic report
            
        Returns:
            SourCandyResponse with final content and diagnostics
            
        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If pipeline execution fails
        '''
        if not isinstance(prompt, str):
            raise TypeError(f'prompt must be a string, got {type(prompt).__name__}')
        if not prompt.strip():
            raise ValueError('prompt cannot be empty')
        
        try:
            rawOutput = self.callBaseLlm(prompt)
            
            sournessMap = self.sournessCalculator.generateSournessMap(
                outputText=rawOutput,
                promptText=prompt
            )
            
            identifiedHallucinations = []
            if enableVerification:
                identifiedHallucinations = self.verifier.getAggregatedHallucinations(
                    rawOutput,
                    sournessMap
                )
                
            finalContent = rawOutput
            if enableCorrection:
                finalContent = self.surgicalFixer.fixSourTokens(
                    originalText=rawOutput,
                    sournessMap=sournessMap,
                    promptContext=prompt
                )
                
            optimizationSuggestion = None
            if returnDiagnostics:
                optimizationSuggestion = self.generateOptimizationSuggestion(
                    sournessMap,
                    identifiedHallucinations
                )
                
            isSweet = (
                sournessMap.aggregateSournessScore < self.sournessThreshold
                and len(identifiedHallucinations) == 0
            )
            
            diagnosticReport = DiagnosticReport(
                rawOutput=rawOutput,
                sournessMap=sournessMap,
                identifiedHallucinations=identifiedHallucinations,
                optimizationSuggestion=optimizationSuggestion
            )
            
            self.runHistory.append({
                'prompt': prompt,
                'rawOutput': rawOutput,
                'finalContent': finalContent,
                'aggregateSourness': sournessMap.aggregateSournessScore,
                'hallucinationCount': len(identifiedHallucinations),
                'isSweet': isSweet
            })
            
            return SourCandyResponse(
                finalContent=finalContent,
                isSweet=isSweet,
                diagnosticReport=diagnosticReport
            )
        except Exception as e:
            raise RuntimeError(f'Pipeline execution failed: {str(e)}')
        
    def callBaseLlm(self, prompt: str) -> str:
        '''
        Invoke the base LLM and extract text response.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text from LLM
        '''
        try:
            if hasattr(self.baseLlm, 'invoke'):
                response = self.baseLlm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            elif callable(self.baseLlm):
                return str(self.baseLlm(prompt))
            else:
                raise ValueError('baseLlm must be callable or have invoke method')
        except Exception as e:
            raise RuntimeError(f'Failed to invoke base LLM: {str(e)}')
            
    def generateOptimizationSuggestion(
        self,
        sournessMap: SournessMap,
        hallucinations: List[str]
    ) -> Optional[str]:
        '''
        Generate actionable optimization suggestions based on diagnostics.
        
        Args:
            sournessMap: Token sourness diagnostics
            hallucinations: List of identified hallucinations
            
        Returns:
            Optimization suggestion string
        '''
        suggestions = []
        
        if sournessMap.aggregateSournessScore > 0.7:
            suggestions.append(
                'High aggregate sourness detected. Consider expanding knowledge base coverage.'
            )
            
        if len(hallucinations) > 3:
            suggestions.append(
                f'Multiple hallucinations found ({len(hallucinations)}). Review domain verifier constraints.'
            )
            
        highSournessTokens = [
            d.token for d in sournessMap.tokenDiagnostics 
            if d.sournessScore > 0.8
        ]
        
        if len(highSournessTokens) > 5:
            suggestions.append(
                f'Critical sourness in tokens: {", ".join(highSournessTokens[:5])}. Add these entities to knowledge graph.'
            )
            
        if not suggestions:
            return 'Output quality is acceptable. No major optimizations needed.'
            
        return ' '.join(suggestions)
    
    def getMetrics(self, domain: Optional[str] = None) -> Dict[str, Any]:
        '''
        Get aggregate metrics across all runs or filtered by domain.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            Dictionary containing performance metrics
        '''
        if not self.runHistory:
            return {
                'totalRuns': 0,
                'averageSourness': 0.0,
                'sweetRate': 0.0,
                'totalHallucinations': 0
            }
            
        totalRuns = len(self.runHistory)
        averageSourness = sum(r['aggregateSourness'] for r in self.runHistory) / totalRuns
        sweetCount = sum(1 for r in self.runHistory if r['isSweet'])
        sweetRate = sweetCount / totalRuns
        totalHallucinations = sum(r['hallucinationCount'] for r in self.runHistory)
        
        return {
            'totalRuns': totalRuns,
            'averageSourness': averageSourness,
            'sweetRate': sweetRate,
            'sweetCount': sweetCount,
            'totalHallucinations': totalHallucinations,
            'averageHallucinationsPerRun': totalHallucinations / totalRuns
        }
    
    def refinePrompt(self, prompt: str, diagnosticReport: DiagnosticReport) -> str:
        '''
        Generate optimized prompt based on diagnostic feedback.
        
        Args:
            prompt: Original prompt
            diagnosticReport: Diagnostic report from previous run
            
        Returns:
            Refined prompt with hallucination prevention guidance
        '''
        return self.promptRefiner.generateOptimizedPrompt(
            originalPrompt=prompt,
            diagnosticReport=diagnosticReport
        )
    
    def batchInvoke(
        self,
        prompts: List[str],
        enableCorrection: bool = True,
        enableVerification: bool = True
    ) -> List[SourCandyResponse]:
        '''
        Process multiple prompts through the pipeline efficiently.
        
        Args:
            prompts: List of input prompts
            enableCorrection: Whether to apply corrections
            enableVerification: Whether to verify outputs
            
        Returns:
            List of SourCandyResponse objects
        '''
        return [
            self.invoke(
                prompt=prompt,
                enableCorrection=enableCorrection,
                enableVerification=enableVerification
            )
            for prompt in prompts
        ]
    
    def resetHistory(self):
        '''
        Clear run history and accumulated statistics.
        '''
        self.runHistory = []
