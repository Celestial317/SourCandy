from typing import List, Dict, Set, Optional, Tuple
import numpy as np


class KnowledgeGraph:
    '''
    Neurosymbolic knowledge graph for entity and relation management.
    Stores entities with vector embeddings and maintains relational structure.
    Supports traversal and subgraph extraction for token grounding operations.
    '''
    
    DEFAULT_MAX_DEPTH = 2
    DEFAULT_MAX_HOPS = 3
    
    def __init__(self):
        '''
        Initialize empty knowledge graph with entity and relation storage.
        Uses adjacency list representation for efficient graph operations.
        '''
        self.entities: Dict[str, np.ndarray] = {}
        self.entityLabels: Dict[str, str] = {}
        self.relations: Dict[str, List[Tuple[str, str]]] = {}
        self.reverseRelations: Dict[str, List[Tuple[str, str]]] = {}
        self.adjacencyList: Dict[str, Set[str]] = {}
        self.embeddingDimension: Optional[int] = None
        
    def addEntity(self, entityId: str, embedding: np.ndarray, label: Optional[str] = None):
        '''
        Add or update an entity in the knowledge graph.
        
        Args:
            entityId: Unique identifier for the entity
            embedding: Vector representation of the entity
            label: Optional human-readable label
            
        Raises:
            TypeError: If parameters have wrong types
            ValueError: If embedding has inconsistent dimension
        '''
        if not isinstance(entityId, str) or not entityId.strip():
            raise ValueError('entityId must be a non-empty string')
        if not isinstance(embedding, np.ndarray):
            raise TypeError('embedding must be a numpy array')
        if embedding.ndim != 1:
            raise ValueError(f'embedding must be 1-dimensional, got shape {embedding.shape}')
        
        embeddingArray = np.array(embedding, dtype=np.float32)
        
        if self.embeddingDimension is None:
            self.embeddingDimension = len(embeddingArray)
        elif len(embeddingArray) != self.embeddingDimension:
            raise ValueError(
                f'Embedding dimension mismatch: expected {self.embeddingDimension}, '
                f'got {len(embeddingArray)} for entity {entityId}'
            )
        
        self.entities[entityId] = embeddingArray
        self.entityLabels[entityId] = label if label else entityId
        if entityId not in self.adjacencyList:
            self.adjacencyList[entityId] = set()
            
    def addRelation(self, subject: str, predicate: str, objectEntity: str):
        '''
        Add a directed relation (subject-predicate-object triplet).
        Automatically creates adjacency connections for graph traversal.
        
        Args:
            subject: Source entity identifier
            predicate: Relation type label
            objectEntity: Target entity identifier
        '''
        if subject not in self.entities or objectEntity not in self.entities:
            raise ValueError('Both subject and object must be added as entities first')
            
        if predicate not in self.relations:
            self.relations[predicate] = []
            
        if predicate not in self.reverseRelations:
            self.reverseRelations[predicate] = []
            
        self.relations[predicate].append((subject, objectEntity))
        self.reverseRelations[predicate].append((objectEntity, subject))
        
        self.adjacencyList[subject].add(objectEntity)
        if objectEntity not in self.adjacencyList:
            self.adjacencyList[objectEntity] = set()
        self.adjacencyList[objectEntity].add(subject)
        
    def getRelatedSubGraph(self, entityIds: List[str], maxDepth: int = 2) -> Dict[str, np.ndarray]:
        '''
        Extract subgraph containing specified entities and their neighbors.
        Used to construct K_G matrix for token grounding alignment.
        
        Args:
            entityIds: List of entity identifiers to include
            maxDepth: Maximum traversal depth from seed entities
            
        Returns:
            Dictionary mapping entity IDs to their embeddings
        '''
        subgraphEntities = set(entityIds)
        queue = [(eid, 0) for eid in entityIds if eid in self.entities]
        visited = set()
        
        while queue:
            currentEntity, depth = queue.pop(0)
            if currentEntity in visited or depth > maxDepth:
                continue
                
            visited.add(currentEntity)
            subgraphEntities.add(currentEntity)
            
            if currentEntity in self.adjacencyList:
                for neighbor in self.adjacencyList[currentEntity]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                        
        return {eid: self.entities[eid] for eid in subgraphEntities if eid in self.entities}
    
    def getEmbeddingMatrix(self, entityIds: Optional[List[str]] = None) -> np.ndarray:
        '''
        Construct embedding matrix K_G for specified entities.
        
        Args:
            entityIds: Optional list of entity IDs. If None, returns all entities
            
        Returns:
            2D array where each row is an entity embedding.
            Returns empty array with proper shape if no entities found.
            
        Raises:
            ValueError: If entityIds contains non-existent entities
        '''
        if entityIds is None:
            entityIds = list(self.entities.keys())
        elif not isinstance(entityIds, list):
            raise TypeError('entityIds must be a list or None')
            
        if entityIds:
            missingIds = [eid for eid in entityIds if eid not in self.entities]
            if missingIds:
                raise ValueError(f'Entity IDs not found in graph: {missingIds[:5]}')
        
        embeddings = [self.entities[eid] for eid in entityIds if eid in self.entities]
        
        if not embeddings:
            if self.embeddingDimension is not None:
                return np.array([]).reshape(0, self.embeddingDimension)
            return np.array([]).reshape(0, 0)
            
        return np.vstack(embeddings)
    
    def hasPath(self, fromEntity: str, toEntity: str, maxHops: int = 3) -> bool:
        '''
        Check if there exists a path between two entities.
        Used for computing R_t relational connectivity score.
        
        Args:
            fromEntity: Source entity identifier
            toEntity: Target entity identifier
            maxHops: Maximum number of hops to search (must be > 0)
            
        Returns:
            True if path exists within maxHops distance
            
        Raises:
            ValueError: If maxHops is invalid or entities don't exist
        '''
        if not isinstance(maxHops, int) or maxHops <= 0:
            raise ValueError(f'maxHops must be a positive integer, got {maxHops}')
        if not isinstance(fromEntity, str) or not isinstance(toEntity, str):
            raise TypeError('Entity IDs must be strings')
        
        if fromEntity not in self.adjacencyList:
            return False
        if toEntity not in self.adjacencyList:
            return False
            
        if fromEntity == toEntity:
            return True
            
        try:
            visited = set()
            queue = [(fromEntity, 0)]
            
            while queue:
                current, hops = queue.pop(0)
                if current == toEntity:
                    return True
                    
                if hops >= maxHops or current in visited:
                    continue
                    
                visited.add(current)
                
                for neighbor in self.adjacencyList[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, hops + 1))
                        
            return False
        except Exception as e:
            raise RuntimeError(f'Error during path search: {str(e)}')
    
    def findClosestEntity(self, queryEmbedding: np.ndarray, topK: int = 5) -> List[Tuple[str, float]]:
        '''
        Find entities most similar to query embedding using cosine similarity.
        
        Args:
            queryEmbedding: Query vector to match against
            topK: Number of top matches to return (must be > 0)
            
        Returns:
            List of (entity_id, similarity_score) tuples sorted by score
            
        Raises:
            TypeError: If queryEmbedding is not a numpy array
            ValueError: If topK is invalid or embedding dimension mismatch
        '''
        if not isinstance(queryEmbedding, np.ndarray):
            raise TypeError('queryEmbedding must be a numpy array')
        if not isinstance(topK, int) or topK <= 0:
            raise ValueError(f'topK must be a positive integer, got {topK}')
        
        if not self.entities:
            return []
        
        if self.embeddingDimension is not None and len(queryEmbedding) != self.embeddingDimension:
            raise ValueError(
                f'Query embedding dimension mismatch: expected {self.embeddingDimension}, '
                f'got {len(queryEmbedding)}'
            )
            
        try:
            queryNorm = np.linalg.norm(queryEmbedding)
            if queryNorm == 0:
                return []
                
            queryEmbedding = queryEmbedding / queryNorm
            
            scores = []
            for entityId, embedding in self.entities.items():
                embeddingNorm = np.linalg.norm(embedding)
                if embeddingNorm > 0:
                    similarity = np.dot(queryEmbedding, embedding / embeddingNorm)
                    scores.append((entityId, float(similarity)))
                    
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:topK]
        except Exception as e:
            raise RuntimeError(f'Error finding closest entities: {str(e)}')
