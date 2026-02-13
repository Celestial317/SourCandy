from typing import List, Dict, Optional
import numpy as np
from core.embedding import BaseEmbedder


class KnowledgeStore:
    '''
    Vector store for efficient similarity search and retrieval.
    Manages document embeddings and provides cosine similarity matching.
    Used to ground generated tokens against source knowledge.
    '''
    
    DEFAULT_TOP_K = 5
    MIN_SIMILARITY_THRESHOLD = -1.0
    MAX_SIMILARITY_THRESHOLD = 1.0
    
    def __init__(self, embedder: BaseEmbedder):
        '''
        Initialize knowledge store with an embedding model.
        
        Args:
            embedder: Instance implementing BaseEmbedder interface
            
        Raises:
            TypeError: If embedder is not a BaseEmbedder instance
        '''
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError('embedder must be a BaseEmbedder instance')
        
        self.embedder = embedder
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        
    def addDocuments(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        '''
        Add documents to the store and compute their embeddings.
        
        Args:
            documents: List of text documents to index
            metadata: Optional metadata for each document
            
        Raises:
            TypeError: If documents is not a list
            ValueError: If documents is empty or metadata length mismatch
            RuntimeError: If embedding computation fails
        '''
        if not isinstance(documents, list):
            raise TypeError(f'documents must be a list, got {type(documents).__name__}')
        if not documents:
            return
        if not all(isinstance(doc, str) for doc in documents):
            raise TypeError('All documents must be strings')
        if metadata is not None:
            if not isinstance(metadata, list):
                raise TypeError('metadata must be a list or None')
            if len(metadata) != len(documents):
                raise ValueError(
                    f'Metadata length ({len(metadata)}) must match documents length ({len(documents)})'
                )
        
        try:
            newEmbeddings = self.embedder.embedBatch(documents)
        except Exception as e:
            raise RuntimeError(f'Failed to compute document embeddings: {str(e)}')
        
        if self.embeddings is None:
            self.embeddings = newEmbeddings
        else:
            self.embeddings = np.vstack([self.embeddings, newEmbeddings])
            
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
            
    def searchSimilar(self, query: str, topK: int = 5) -> List[Dict]:
        '''
        Find most similar documents to query string.
        
        Args:
            query: Query text to search for
            topK: Number of top results to return (must be > 0)
            
        Returns:
            List of dictionaries with document, score, and metadata
            
        Raises:
            TypeError: If query is not a string or topK is not an integer
            ValueError: If topK is invalid
            RuntimeError: If search fails
        '''
        if not isinstance(query, str):
            raise TypeError(f'query must be a string, got {type(query).__name__}')
        if not isinstance(topK, int) or topK <= 0:
            raise ValueError(f'topK must be a positive integer, got {topK}')
        
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        try:
            queryEmbedding = self.embedder.embedText(query)
        except Exception as e:
            raise RuntimeError(f'Failed to embed query: {str(e)}')
        
        try:
            similarities = self.cosineSimilarity(queryEmbedding, self.embeddings)
            
            topIndices = np.argsort(similarities)[::-1][:topK]
            
            results = []
            for idx in topIndices:
                results.append({
                    'document': self.documents[idx],
                    'score': float(similarities[idx]),
                    'metadata': self.metadata[idx],
                    'index': int(idx)
                })
                
            return results
        except Exception as e:
            raise RuntimeError(f'Failed to search similar documents: {str(e)}')
    
    def searchByEmbedding(self, embedding: np.ndarray, topK: int = 5) -> List[int]:
        '''
        Find documents most similar to given embedding vector.
        
        Args:
            embedding: Query embedding vector
            topK: Number of results to return
            
        Returns:
            List of document indices sorted by similarity
        '''
        if self.embeddings is None:
            return []
            
        similarities = self.cosineSimilarity(embedding, self.embeddings)
        topIndices = np.argsort(similarities)[::-1][:topK]
        
        return topIndices.tolist()
    
    @staticmethod
    def cosineSimilarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        '''
        Compute cosine similarity between vector and each row in matrix.
        
        Args:
            vector: Query vector of shape (d,)
            matrix: Matrix of shape (n, d)
            
        Returns:
            Array of similarity scores of shape (n,)
        '''
        vectorNorm = np.linalg.norm(vector)
        if vectorNorm == 0:
            return np.zeros(matrix.shape[0])
            
        normalizedVector = vector / vectorNorm
        
        matrixNorms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrixNorms[matrixNorms == 0] = 1
        normalizedMatrix = matrix / matrixNorms
        
        return np.dot(normalizedMatrix, normalizedVector)
    
    def getEmbedding(self, index: int) -> Optional[np.ndarray]:
        '''
        Retrieve embedding vector for document at given index.
        
        Args:
            index: Document index
            
        Returns:
            Embedding vector or None if index invalid
        '''
        if self.embeddings is None or index >= len(self.documents):
            return None
            
        return self.embeddings[index]
    
    def clear(self):
        '''
        Remove all documents and embeddings from the store.
        '''
        self.documents = []
        self.embeddings = None
        self.metadata = []
