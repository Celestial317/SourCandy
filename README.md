# SourCandy 🍬

**Token-Level Hallucination Detection and Surgical Correction for LLMs**

SourCandy is a production-grade framework that transforms unreliable LLM outputs into factually grounded responses through granular, token-level diagnostic analysis and surgical correction. Unlike traditional approaches that evaluate at the sentence or document level, SourCandy audits **every single token** using a novel Modified Cross Attention scoring mechanism.

## 🎯 Core Innovation

SourCandy implements a research-driven **Sourness Index** that scores each token using:

```
S_t = 1 - tanh(w₁·G_t + w₂·P_t + w₃·R_t)
```

Where:
- **G_t** (Knowledge Grounding): Max softmax alignment between token and knowledge base
- **P_t** (Prompt Correspondence): Alignment between token and input prompt  
- **R_t** (Relational Connectivity): Neurosymbolic graph path verification

## 🚀 Key Features

- **Token-Level Diagnostics**: Granular sourness scores for every generated token
- **Surgical Correction**: Replace only hallucinated tokens while preserving context
- **Knowledge Graph Integration**: Neurosymbolic grounding with entity relations
- **Domain Verification**: Specialized validators for different knowledge domains
- **Prompt Optimization**: Auto-generate improved prompts from diagnostic feedback
- **Visual Debugging**: Heatmaps and distributions for sourness analysis
- **Production Ready**: Type-safe, well-documented, zero-comment clean code

## 📦 Installation

```bash
# Basic installation
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With all optional dependencies
pip install -e ".[all]"
```

## 🎓 Quick Start

```python
from sourcandy import Guard, KnowledgeGraph, SimpleEmbedder

# Initialize components
embedder = SimpleEmbedder()
knowledgeGraph = KnowledgeGraph()

# Populate knowledge graph
knowledgeGraph.addEntity(
    entityId='python',
    embedding=embedder.embedText('Python programming language'),
    label='Python'
)

# Create Guard instance
guard = Guard(
    baseLlm=yourLlm,  # Any LLM with .invoke() method
    knowledgeGraph=knowledgeGraph,
    embedder=embedder,
    sournessThreshold=0.6
)

# Add knowledge documents
guard.addKnowledgeDocuments([
    'Python was created by Guido van Rossum in 1991.',
    'Python is used for data science and web development.'
])

# Process query
response = guard.invoke('Tell me about Python')

print(f'Original: {response.diagnosticReport.rawOutput}')
print(f'Corrected: {response.finalContent}')
print(f'Is Sweet: {response.isSweet}')
print(f'Sourness: {response.diagnosticReport.sournessMap.aggregateSournessScore:.3f}')
```

## 🏗️ Architecture

### Pipeline Flow

```
Input Prompt
    ↓
Knowledge Retrieval
    ↓
Base LLM Call → Raw Output
    ↓
Token-Level Sourness Calculation
    ↓
Domain Verification
    ↓
Surgical Token Correction
    ↓
Final Sweet Output + Diagnostics
```

### Core Components

1. **Guard**: Main orchestrator managing the entire pipeline
2. **SournessCalculator**: Implements the Modified Cross Attention formula
3. **KnowledgeGraph**: Neurosymbolic graph for entity grounding and path verification
4. **KnowledgeStore**: Vector store for document similarity matching
5. **SurgicalFixer**: Token-level correction with LLM-based infilling
6. **DomainVerifier**: Specialized validators for domain-specific constraints
7. **PromptRefiner**: Auto-generates optimized prompts from diagnostics

## 📊 Visualization

```python
from sourcandy import plotSournessHeatmap

response = guard.invoke('Your query here')

# Generate heatmap
plotSournessHeatmap(
    response.diagnosticReport.sournessMap,
    title='Token Sourness Analysis',
    savePath='heatmap.png'
)
```

## 🔬 Advanced Usage

### Custom Embedders

```python
from sourcandy import BaseEmbedder
import numpy as np

class CustomEmbedder(BaseEmbedder):
    def embedText(self, text: str) -> np.ndarray:
        # Your embedding logic
        return np.array([...])
    
    def embedBatch(self, texts: list) -> np.ndarray:
        return np.vstack([self.embedText(t) for t in texts])
```

### Domain Verification

```python
from sourcandy import DomainVerifier

# Create domain verifier
medicalVerifier = DomainVerifier(
    domain='medical',
    knowledgeGraph=knowledgeGraph,
    knowledgeStore=knowledgeStore,
    embedder=embedder
)

# Add domain keywords
medicalVerifier.addDomainKeywords([
    'diagnosis', 'treatment', 'symptoms', 'medication'
])

# Add constraints
medicalVerifier.addConstraint(
    'forbidden',
    r'guaranteed cure|100% effective'
)

# Register with guard
guard.addDomainVerifier('medical', medicalVerifier)
```

### Metrics Tracking

```python
# Process multiple queries
for prompt in prompts:
    response = guard.invoke(prompt)

# Get aggregate metrics
metrics = guard.getMetrics()
print(f"Sweet Rate: {metrics['sweetRate']:.1%}")
print(f"Average Sourness: {metrics['averageSourness']:.3f}")
```

### Prompt Refinement

```python
# Initial query
response = guard.invoke('What are the latest Python features?')

# Generate optimized prompt
refinedPrompt = guard.refinePrompt(
    prompt='What are the latest Python features?',
    diagnosticReport=response.diagnosticReport
)

# Use refined prompt for better results
betterResponse = guard.invoke(refinedPrompt)
```

## 🎨 Framework Design Principles

- **camelCase**: All identifiers use camelCase convention
- **Single Quotes**: All strings use single quotes
- **No Comments**: Self-documenting code with comprehensive docstrings
- **Type Safety**: Pydantic models for all data structures
- **Production Grade**: Error handling, validation, efficient algorithms

## 📖 API Reference

### Guard

Main orchestrator class.

**Methods:**
- `invoke(prompt, enableCorrection=True, enableVerification=True)`: Process single query
- `batchInvoke(prompts, ...)`: Process multiple queries
- `addKnowledgeDocuments(documents, metadata)`: Add factual documents
- `addDomainVerifier(domain, verifier)`: Register domain validator
- `getMetrics()`: Get performance statistics
- `refinePrompt(prompt, diagnosticReport)`: Generate optimized prompt

### SournessCalculator

Token-level sourness scoring.

**Methods:**
- `calcG(tokenEmbedding, kgMatrix)`: Calculate knowledge grounding score
- `calcP(tokenEmbedding, promptEmbedding)`: Calculate prompt correspondence
- `calcR(currentGrounding, previousGrounding)`: Calculate relational connectivity
- `generateSournessMap(outputText, promptText)`: Generate full diagnostic map

### KnowledgeGraph

Neurosymbolic entity graph.

**Methods:**
- `addEntity(entityId, embedding, label)`: Add entity with vector
- `addRelation(subject, predicate, object)`: Add triplet relation
- `getRelatedSubGraph(entityIds, maxDepth)`: Extract subgraph
- `hasPath(fromEntity, toEntity, maxHops)`: Check connectivity
- `findClosestEntity(queryEmbedding, topK)`: Similarity search

## 🧪 Examples

See [examples.py](examples.py) for comprehensive demonstrations including:
- Basic usage with hallucination detection
- Sourness visualization
- Metrics tracking across multiple queries  
- Automatic prompt refinement

Run examples:
```bash
python examples.py
```

## 🤝 Contributing

We welcome contributions! Please ensure:
- Follow camelCase naming conventions
- Use single quotes for strings
- Provide comprehensive docstrings (no inline comments)
- Add type hints to all functions
- Include tests for new features

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

Built with:
- **Pydantic**: Data validation and settings management
- **NumPy**: Efficient numerical operations
- **Matplotlib/Seaborn**: Visualization (optional)

## 📧 Contact

- **Authors**: Soumya Sourav Das, Devyanshi Bansal
- **Issues**: [GitHub Issues](https://github.com/celestial317/sourcandy/issues)

---

**Make your LLM outputs sweet, not sour! 🍬**
