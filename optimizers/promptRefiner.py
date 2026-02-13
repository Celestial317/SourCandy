from typing import List, Dict, Any, Optional
from schema import DiagnosticReport, SournessMap, TokenDiagnosis
from collections import defaultdict


BaseLLM = Any


class PromptRefiner:
    '''
    Automatically generates optimized system prompts based on diagnostic reports.
    Analyzes patterns in identified hallucinations and sourness distributions
    to prevent recurring errors in future LLM runs.
    '''
    
    DEFAULT_SOUR_THRESHOLD = 0.6
    DEFAULT_AVERAGE_SOURNESS_THRESHOLD = 0.5
    DEFAULT_MAX_PROBLEMATIC_TOKENS = 5
    DEFAULT_MAX_HALLUCINATION_EXAMPLES = 3
    
    def __init__(
        self,
        baseLlm: Optional[BaseLLM] = None,
        sourThreshold: float = 0.6,
        avgSournessThreshold: float = 0.5,
        maxProblematicTokens: int = 5,
        maxHallucinationExamples: int = 3
    ):
        '''
        Initialize prompt refiner with optional LLM for advanced refinement.
        
        Args:
            baseLlm: Optional language model for generating refined prompts
            sourThreshold: Threshold for identifying problematic tokens (0.0-1.0)
            avgSournessThreshold: Threshold for average sourness warnings (0.0-1.0)
            maxProblematicTokens: Max number of problematic tokens to report
            maxHallucinationExamples: Max number of hallucination examples to include
            
        Raises:
            ValueError: If thresholds are out of valid range
        '''
        if not (0.0 <= sourThreshold <= 1.0):
            raise ValueError(f'sourThreshold must be between 0.0 and 1.0, got {sourThreshold}')
        if not (0.0 <= avgSournessThreshold <= 1.0):
            raise ValueError(f'avgSournessThreshold must be between 0.0 and 1.0, got {avgSournessThreshold}')
        if not isinstance(maxProblematicTokens, int) or maxProblematicTokens <= 0:
            raise ValueError(f'maxProblematicTokens must be a positive integer, got {maxProblematicTokens}')
        if not isinstance(maxHallucinationExamples, int) or maxHallucinationExamples <= 0:
            raise ValueError(f'maxHallucinationExamples must be a positive integer, got {maxHallucinationExamples}')
        
        self.baseLlm = baseLlm
        self.sourThreshold = sourThreshold
        self.avgSournessThreshold = avgSournessThreshold
        self.maxProblematicTokens = maxProblematicTokens
        self.maxHallucinationExamples = maxHallucinationExamples
        self.hallucinationPatterns: Dict[str, int] = defaultdict(int)
        self.highSournessTokens: Dict[str, List[float]] = defaultdict(list)
        
    def analyzeReport(self, report: DiagnosticReport) -> Dict[str, Any]:
        '''
        Analyze diagnostic report to extract patterns and insights.
        
        Args:
            report: Diagnostic report from SourCandy pipeline
            
        Returns:
            Dictionary containing analysis metrics and patterns
            
        Raises:
            TypeError: If report is not a DiagnosticReport instance
            RuntimeError: If analysis fails
        '''
        if not isinstance(report, DiagnosticReport):
            raise TypeError('report must be a DiagnosticReport instance')
        
        try:
            analysis = {
                'totalTokens': len(report.sournessMap.tokenDiagnostics),
                'averageSourness': report.sournessMap.aggregateSournessScore,
                'hallucinationCount': len(report.identifiedHallucinations),
                'problematicTokens': [],
                'hallucinationTypes': []
            }
            
            for diagnosis in report.sournessMap.tokenDiagnostics:
                if diagnosis.sournessScore > self.sourThreshold:
                    analysis['problematicTokens'].append({
                        'token': diagnosis.token,
                        'score': diagnosis.sournessScore
                    })
                    self.highSournessTokens[diagnosis.token].append(diagnosis.sournessScore)
                    
            for hallucination in report.identifiedHallucinations:
                self.hallucinationPatterns[hallucination] += 1
                analysis['hallucinationTypes'].append(hallucination)
                
            return analysis
        except Exception as e:
            raise RuntimeError(f'Failed to analyze report: {str(e)}')
    
    def generateOptimizedPrompt(
        self,
        originalPrompt: str,
        diagnosticReport: DiagnosticReport,
        domain: Optional[str] = None
    ) -> str:
        '''
        Generate optimized system prompt to prevent identified hallucinations.
        Creates specific instructions based on sourness patterns and error types.
        
        Args:
            originalPrompt: The original user prompt
            diagnosticReport: Full diagnostic report from pipeline
            domain: Optional domain specification for targeted guidance
            
        Returns:
            Optimized system prompt with hallucination prevention instructions
            
        Raises:
            TypeError: If parameters are of wrong type
            RuntimeError: If prompt generation fails
        '''
        if not isinstance(originalPrompt, str):
            raise TypeError(f'originalPrompt must be a string, got {type(originalPrompt).__name__}')
        if not isinstance(diagnosticReport, DiagnosticReport):
            raise TypeError('diagnosticReport must be a DiagnosticReport instance')
        if domain is not None and not isinstance(domain, str):
            raise TypeError(f'domain must be a string or None, got {type(domain).__name__}')
        
        try:
            analysis = self.analyzeReport(diagnosticReport)
            
            systemInstructions = []
            
            if analysis['averageSourness'] > self.avgSournessThreshold:
                systemInstructions.append(
                    'Focus on providing factually grounded responses. Every claim should be traceable to verified knowledge.'
                )
                
            if analysis['hallucinationCount'] > 0:
                systemInstructions.append(
                    f'Avoid making unsupported claims. {analysis["hallucinationCount"]} potential hallucinations were detected in previous responses.'
                )
                
            if analysis['problematicTokens']:
                problematicTerms = [pt['token'] for pt in analysis['problematicTokens'][:self.maxProblematicTokens]]
                systemInstructions.append(
                    f'Exercise caution with terms like: {", ".join(problematicTerms)}. Ensure these are well-grounded.'
                )
                
            if diagnosticReport.identifiedHallucinations:
                specificIssues = diagnosticReport.identifiedHallucinations[:self.maxHallucinationExamples]
                systemInstructions.append(
                    f'Previous errors to avoid: {"; ".join(specificIssues)}'
                )
                
            if domain:
                systemInstructions.append(
                    f'This query is in the {domain} domain. Ensure domain-specific accuracy and terminology.'
                )
                
            if self.baseLlm and len(systemInstructions) > 2:
                optimizedPrompt = self.llmGeneratePrompt(
                    originalPrompt, 
                    systemInstructions,
                    analysis
                )
            else:
                optimizedPrompt = self.templateGeneratePrompt(
                    originalPrompt,
                    systemInstructions
                )
                
            return optimizedPrompt
        except Exception as e:
            raise RuntimeError(f'Failed to generate optimized prompt: {str(e)}')
    
    def templateGeneratePrompt(
        self,
        originalPrompt: str,
        instructions: List[str]
    ) -> str:
        '''
        Generate optimized prompt using template-based approach.
        
        Args:
            originalPrompt: Original user prompt
            instructions: List of system instruction strings
            
        Returns:
            Formatted system + user prompt
        '''
        systemPrompt = 'You are a knowledgeable and precise assistant. Follow these guidelines:\n\n'
        
        for idx, instruction in enumerate(instructions, 1):
            systemPrompt += f'{idx}. {instruction}\n'
            
        systemPrompt += '\nUser Query:\n'
        
        return systemPrompt + originalPrompt
    
    def llmGeneratePrompt(
        self,
        originalPrompt: str,
        instructions: List[str],
        analysis: Dict[str, Any]
    ) -> str:
        '''
        Use LLM to generate sophisticated optimized prompt.
        
        Args:
            originalPrompt: Original user prompt
            instructions: Extracted instruction guidelines
            analysis: Analysis metrics from diagnostic report
            
        Returns:
            LLM-generated optimized prompt
        '''
        metaPrompt = f'''You are an expert prompt engineer. Given the following:

Original User Query: {originalPrompt}

Issues Detected:
- Average Sourness Score: {analysis['averageSourness']:.2f}
- Hallucinations Found: {analysis['hallucinationCount']}
- Problematic Tokens: {len(analysis['problematicTokens'])}

Guidelines to incorporate:
{chr(10).join(f'- {inst}' for inst in instructions)}

Generate an optimized system prompt that will prevent these issues while preserving the user's intent. The prompt should guide the model to be factual, grounded, and precise.

Optimized System Prompt:'''
        
        try:
            if hasattr(self.baseLlm, 'invoke'):
                response = self.baseLlm.invoke(metaPrompt)
                if hasattr(response, 'content'):
                    return response.content.strip() + '\n\n' + originalPrompt
                return str(response).strip() + '\n\n' + originalPrompt
            elif callable(self.baseLlm):
                return str(self.baseLlm(metaPrompt)).strip() + '\n\n' + originalPrompt
            else:
                return self.templateGeneratePrompt(originalPrompt, instructions)
        except Exception:
            return self.templateGeneratePrompt(originalPrompt, instructions)
    
    def batchRefine(
        self,
        prompts: List[str],
        reports: List[DiagnosticReport]
    ) -> List[str]:
        '''
        Refine multiple prompts based on their diagnostic reports.
        
        Args:
            prompts: List of original prompts
            reports: Corresponding diagnostic reports
            
        Returns:
            List of optimized prompts
        '''
        if len(prompts) != len(reports):
            raise ValueError('Prompts and reports lists must have equal length')
            
        return [
            self.generateOptimizedPrompt(prompt, report)
            for prompt, report in zip(prompts, reports)
        ]
    
    def getSummaryStatistics(self) -> Dict[str, Any]:
        '''
        Get accumulated statistics from all analyzed reports.
        
        Returns:
            Dictionary with hallucination patterns and token statistics
        '''
        return {
            'frequentHallucinations': dict(sorted(
                self.hallucinationPatterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'chronicSourTokens': {
                token: {
                    'count': len(scores),
                    'averageScore': sum(scores) / len(scores)
                }
                for token, scores in self.highSournessTokens.items()
                if len(scores) > 2
            }
        }
