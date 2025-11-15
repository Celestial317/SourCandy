from typing import Any, Dict, List
from .interfaces import BaseVerifier, BaseCorrector, SourCandyResponse


# giving users the flexibility to pass in any LLM type
BaseLLM = Any

class Guard:
    """
    The main orchestrator for the SourCandy framework.
    It wraps a base LLM and applies a 'Verify -> Correct -> Re-verify'
    loop based on its configuration.
    """
    
    def __init__(self, base_llm: BaseLLM, config: Dict[str, Any]):
        """
        Initializes the Guard.

        Args:
            base_llm: The core LLM to wrap (e.g., from LangChain, LlamaIndex).
            config: A dictionary defining the Guard's behavior.
                Expected keys:
                - 'verifiers': A list of BaseVerifier instances.
                - 'correctors': A list of BaseCorrector instances.
                - 'log_level': (e.g., 'info', 'debug')
                - 'max_loops': Max number of re-verification loops (default: 3)
        """
        self.base_llm = base_llm
        self.verifiers: List[BaseVerifier] = config.get("verifiers", [])
        self.correctors: List[BaseCorrector] = config.get("correctors", [])
        self.max_loops = config.get("max_loops", 3)
        self.log_level = config.get("log_level", "info")
        
        # TODO: Initialize logging
        print(f"SourCandy Guard initialized with {len(self.verifiers)} verifiers and {len(self.correctors)} correctors.")

    def invoke(self, prompt: str) -> SourCandyResponse:
        """
        Executes the full Guard pipeline:
        1. Calls the base LLM.
        2. Runs the 'Verify -> Correct -> Re-verify' loop.
        3. Returns the final, sweetened response.

        Args:
            prompt: The user's prompt to the LLM.

        Returns:
            A SourCandyResponse object with the final content and report.
        """
        
        # ---
        # This is a STUB for Phase 1.
        # ---
        
        print(f"Guard invoking base LLM with prompt: '{prompt[:50]}...'")
        
        # 1. Call the base LLM
        # We assume the base_llm has an `invoke` method that returns a
        # string or an object with a `content` attribute.
        raw_response = self.base_llm.invoke(prompt)
        
        # Handle different response types (e.g., string vs. AIMessage)
        if hasattr(raw_response, 'content'):
            raw_output = raw_response.content
        else:
            raw_output = str(raw_response)
        
        # TODO (Phase 4): Implement the verification/correction loop here.
        
        final_output = raw_output
        final_report = {
            "run_1": {
                "input": prompt,
                "raw_output": raw_output,
                "verifications": [],
                "corrections": [],
                "status": "Phase 1 Stub: Passthrough"
            }
        }

        return SourCandyResponse(
            content=final_output,
            sour_report=final_report,
            raw_output=raw_output,
            is_final_sour=False # Assume sweet for now
        )