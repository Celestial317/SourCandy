from typing import List
import abc
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    '''
    Abstract base class for text embedding operations.
    Provides interface for converting text into vector representations.
    All embedders must implement embedText and embedBatch methods.
    '''
    
    @abstractmethod
    def embedText(self, text: str) -> np.ndarray:
        '''
        Convert a single text string into an embedding vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array representing the text embedding
        '''
        pass

    @abstractmethod
    def embedBatch(self, texts: List[str]) -> np.ndarray:
        '''
        Convert multiple text strings into embedding vectors.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            2D numpy array where each row is an embedding
        '''
        pass


class SimpleEmbedder(BaseEmbedder):
    '''
    Simple implementation of BaseEmbedder using hash-based projections.
    Provides deterministic embeddings for testing and basic use cases.
    Uses numpy for efficient vector operations and normalization.
    '''
    
    DEFAULT_DIMENSION = 384
    DEFAULT_PROJECTION_SIZE = 1000
    RANDOM_SEED = 3
    
    def __init__(self, modelName: str = 'simple', dimension: int = 384, projectionSize: int = 1000):
        '''
        Initialize the simple embedder.
        
        Args:
            modelName: Name identifier for the embedding model
            dimension: Dimensionality of the output embeddings (must be > 0)
            projectionSize: Size of the hash projection matrix (must be > 0)
            
        Raises:
            ValueError: If dimension or projectionSize is invalid
        '''
        if not isinstance(modelName, str) or not modelName.strip():
            raise ValueError('modelName must be a non-empty string')
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f'dimension must be a positive integer, got {dimension}')
        if not isinstance(projectionSize, int) or projectionSize <= 0:
            raise ValueError(f'projectionSize must be a positive integer, got {projectionSize}')
        
        self.modelName = modelName
        self.dimension = dimension
        self.projectionSize = projectionSize
        
        try:
            np.random.seed(self.RANDOM_SEED)
            self.projectionMatrix = np.random.randn(projectionSize, dimension).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f'Failed to initialize projection matrix: {str(e)}')
        
    def embedText(self, text: str) -> np.ndarray:
        '''
        Generate embedding for single text using hash projection.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized embedding vector of shape (dimension,)
            
        Raises:
            TypeError: If text is not a string
            RuntimeError: If embedding computation fails
        '''
        if not isinstance(text, str):
            raise TypeError(f'text must be a string, got {type(text).__name__}')
        
        try:
            tokens = text.lower().split()
            if not tokens:
                return np.zeros(self.dimension, dtype=np.float32)
                
            embedding = np.zeros(self.dimension, dtype=np.float32)
            for token in tokens:
                hashValue = hash(token) % self.projectionSize
                embedding += self.projectionMatrix[hashValue]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            raise RuntimeError(f'Failed to compute embedding: {str(e)}')

    def embedBatch(self, texts: List[str]) -> np.ndarray:
        '''
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings
            
        Returns:
            2D array of shape (len(texts), dimension)
            
        Raises:
            TypeError: If texts is not a list
            ValueError: If texts list is empty
            RuntimeError: If batch embedding fails
        '''
        if not isinstance(texts, list):
            raise TypeError(f'texts must be a list, got {type(texts).__name__}')
        if not texts:
            raise ValueError('texts list cannot be empty')
        
        try:
            return np.vstack([self.embedText(text) for text in texts])
        except Exception as e:
            raise RuntimeError(f'Failed to compute batch embeddings: {str(e)}')