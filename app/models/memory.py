import os
import logging
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Memory system for storing and retrieving information using vector embeddings.
    
    This class provides long-term memory capabilities for the AI assistant.
    """
    
    def __init__(self):
        """Initialize the memory system with the configured vector database."""
        self.db_type = os.getenv("VECTOR_DB", "chroma")
        self.db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        
        logger.info(f"Initializing MemorySystem with {self.db_type} at {self.db_path}")
        
        # Initialize the vector database
        self._initialize_db()
        
        # Initialize the embedding model
        self._initialize_embedding_model()
    
    def _initialize_db(self):
        """Initialize the vector database based on the configuration."""
        if self.db_type == "chroma":
            self._initialize_chroma_db()
        elif self.db_type == "faiss":
            self._initialize_faiss_db()
        else:
            logger.warning(f"Unknown database type: {self.db_type}, falling back to ChromaDB")
            self.db_type = "chroma"
            self._initialize_chroma_db()
    
    def _initialize_chroma_db(self):
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure the database directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.db_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get the collection
            self.collection = self.db_client.get_or_create_collection(
                name="intelassist_memory",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized ChromaDB at {self.db_path}")
        except ImportError:
            logger.error("Failed to import ChromaDB libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def _initialize_faiss_db(self):
        """Initialize FAISS."""
        try:
            import faiss
            import pickle
            import os.path
            
            # Ensure the database directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Path to the index file
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            metadata_path = os.path.join(self.db_path, "faiss_metadata.pkl")
            
            # Check if the index already exists
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load existing index
                self.index = faiss.read_index(index_path)
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
            else:
                # Create a new index
                embedding_dim = 768  # Default for sentence-transformers
                self.index = faiss.IndexFlatL2(embedding_dim)
                self.metadata = {
                    "ids": [],
                    "contents": [],
                    "user_ids": [],
                    "timestamps": [],
                    "additional_metadata": []
                }
            
            logger.info(f"Initialized FAISS at {self.db_path}")
        except ImportError:
            logger.error("Failed to import FAISS libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for text."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize the embedding model
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Initialized embedding model: all-MiniLM-L6-v2")
        except ImportError:
            logger.error("Failed to import SentenceTransformer libraries")
            raise
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def store(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Store a memory entry in the vector database.
        
        Args:
            content: The content to store
            metadata: Additional metadata for the entry
            user_id: User ID associated with the memory
            
        Returns:
            ID of the stored memory
        """
        logger.info(f"Storing memory: {content[:50]}...")
        
        try:
            # Generate a unique ID
            memory_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self._get_embedding(content)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            full_metadata = {
                "timestamp": time.time(),
                "user_id": user_id,
                **metadata
            }
            
            # Store in the appropriate database
            if self.db_type == "chroma":
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[full_metadata],
                    documents=[content]
                )
            
            elif self.db_type == "faiss":
                # Add to FAISS index
                self.index.add(np.array([embedding], dtype=np.float32))
                
                # Update metadata
                self.metadata["ids"].append(memory_id)
                self.metadata["contents"].append(content)
                self.metadata["user_ids"].append(user_id)
                self.metadata["timestamps"].append(time.time())
                self.metadata["additional_metadata"].append(metadata)
                
                # Save the updated index and metadata
                faiss.write_index(self.index, os.path.join(self.db_path, "faiss_index.bin"))
                with open(os.path.join(self.db_path, "faiss_metadata.pkl"), "wb") as f:
                    import pickle
                    pickle.dump(self.metadata, f)
            
            logger.info(f"Stored memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.7
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search for relevant memories based on a query.
        
        Args:
            query: The search query
            user_id: Filter results by user ID
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            Tuple containing:
            - results: List of matching memories
            - total: Total number of matching results
        """
        logger.info(f"Searching memory with query: {query[:50]}...")
        
        try:
            # Generate embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Search in the appropriate database
            if self.db_type == "chroma":
                # Prepare filter
                where_clause = None
                if user_id:
                    where_clause = {"user_id": user_id}
                
                # Search in ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause
                )
                
                # Process results
                processed_results = []
                for i in range(len(results["ids"][0])):
                    # Skip results below the threshold
                    if results["distances"][0][i] > 1 - min_score:
                        continue
                    
                    processed_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    })
                
                total = len(processed_results)
            
            elif self.db_type == "faiss":
                # Search in FAISS
                k = min(limit * 2, len(self.metadata["ids"]))  # Get more results for filtering
                if k == 0:
                    return [], 0
                
                distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
                
                # Process results
                processed_results = []
                for i, idx in enumerate(indices[0]):
                    # Skip results below the threshold
                    similarity = 1 - distances[0][i] / 2  # Convert L2 distance to similarity (simplified)
                    if similarity < min_score:
                        continue
                    
                    # Skip results from other users if user_id is specified
                    if user_id and self.metadata["user_ids"][idx] != user_id:
                        continue
                    
                    processed_results.append({
                        "id": self.metadata["ids"][idx],
                        "content": self.metadata["contents"][idx],
                        "metadata": {
                            "user_id": self.metadata["user_ids"][idx],
                            "timestamp": self.metadata["timestamps"][idx],
                            **self.metadata["additional_metadata"][idx]
                        },
                        "score": similarity
                    })
                
                # Limit results
                processed_results = processed_results[:limit]
                total = len(processed_results)
            
            logger.info(f"Found {total} relevant memories")
            return processed_results, total
            
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            raise
    
    def delete(
        self,
        memory_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete a specific memory entry by ID.
        
        Args:
            memory_id: ID of the memory to delete
            user_id: User ID for verification
            
        Returns:
            True if the memory was deleted, False otherwise
        """
        logger.info(f"Deleting memory with ID: {memory_id}")
        
        try:
            # Delete from the appropriate database
            if self.db_type == "chroma":
                # Prepare filter
                where_clause = None
                if user_id:
                    where_clause = {"user_id": user_id}
                
                # Get the memory to verify it exists
                results = self.collection.get(
                    ids=[memory_id],
                    where=where_clause
                )
                
                if not results["ids"]:
                    logger.warning(f"Memory with ID {memory_id} not found")
                    return False
                
                # Delete the memory
                self.collection.delete(
                    ids=[memory_id],
                    where=where_clause
                )
            
            elif self.db_type == "faiss":
                # Find the memory index
                try:
                    idx = self.metadata["ids"].index(memory_id)
                except ValueError:
                    logger.warning(f"Memory with ID {memory_id} not found")
                    return False
                
                # Verify user_id if specified
                if user_id and self.metadata["user_ids"][idx] != user_id:
                    logger.warning(f"Memory with ID {memory_id} does not belong to user {user_id}")
                    return False
                
                # FAISS doesn't support direct deletion, so we need to rebuild the index
                # This is inefficient for frequent deletions
                new_index = faiss.IndexFlatL2(self.index.d)
                
                # Copy all embeddings except the one to delete
                embeddings = []
                for i in range(self.index.ntotal):
                    if i != idx:
                        embedding = np.array([self.index.reconstruct(i)], dtype=np.float32)
                        embeddings.append(embedding)
                
                # Update metadata
                for key in ["ids", "contents", "user_ids", "timestamps", "additional_metadata"]:
                    self.metadata[key] = [
                        self.metadata[key][i] for i in range(len(self.metadata[key])) if i != idx
                    ]
                
                # Add all embeddings to the new index
                if embeddings:
                    embeddings = np.vstack(embeddings)
                    new_index.add(embeddings)
                
                # Replace the old index
                self.index = new_index
                
                # Save the updated index and metadata
                faiss.write_index(self.index, os.path.join(self.db_path, "faiss_index.bin"))
                with open(os.path.join(self.db_path, "faiss_metadata.pkl"), "wb") as f:
                    import pickle
                    pickle.dump(self.metadata, f)
            
            logger.info(f"Deleted memory with ID: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            raise
    
    def batch_store(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple memory entries in a single batch operation.
        
        Args:
            entries: List of memory entries to store
            
        Returns:
            List of IDs for the stored memories
        """
        logger.info(f"Batch storing {len(entries)} memories")
        
        try:
            memory_ids = []
            
            if self.db_type == "chroma":
                # Prepare batch data
                ids = []
                embeddings = []
                metadatas = []
                documents = []
                
                for entry in entries:
                    # Generate a unique ID
                    memory_id = str(uuid.uuid4())
                    memory_ids.append(memory_id)
                    
                    # Extract entry data
                    content = entry["content"]
                    metadata = entry.get("metadata", {})
                    user_id = entry.get("user_id")
                    
                    # Generate embedding
                    embedding = self._get_embedding(content)
                    
                    # Prepare metadata
                    full_metadata = {
                        "timestamp": time.time(),
                        "user_id": user_id,
                        **metadata
                    }
                    
                    # Add to batch
                    ids.append(memory_id)
                    embeddings.append(embedding)
                    metadatas.append(full_metadata)
                    documents.append(content)
                
                # Store the batch
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            
            elif self.db_type == "faiss":
                # Prepare batch data
                embeddings = []
                
                for entry in entries:
                    # Generate a unique ID
                    memory_id = str(uuid.uuid4())
                    memory_ids.append(memory_id)
                    
                    # Extract entry data
                    content = entry["content"]
                    metadata = entry.get("metadata", {})
                    user_id = entry.get("user_id")
                    
                    # Generate embedding
                    embedding = self._get_embedding(content)
                    embeddings.append(embedding)
                    
                    # Update metadata
                    self.metadata["ids"].append(memory_id)
                    self.metadata["contents"].append(content)
                    self.metadata["user_ids"].append(user_id)
                    self.metadata["timestamps"].append(time.time())
                    self.metadata["additional_metadata"].append(metadata)
                
                # Add to FAISS index
                self.index.add(np.array(embeddings, dtype=np.float32))
                
                # Save the updated index and metadata
                faiss.write_index(self.index, os.path.join(self.db_path, "faiss_index.bin"))
                with open(os.path.join(self.db_path, "faiss_metadata.pkl"), "wb") as f:
                    import pickle
                    pickle.dump(self.metadata, f)
            
            logger.info(f"Batch stored {len(memory_ids)} memories")
            return memory_ids
            
        except Exception as e:
            logger.error(f"Error batch storing memories: {str(e)}")
            raise
    
    def get_stats(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Args:
            user_id: Filter stats by user ID
            
        Returns:
            Dictionary with memory statistics
        """
        logger.info(f"Getting memory stats for user_id: {user_id}")
        
        try:
            stats = {
                "db_type": self.db_type,
                "db_path": self.db_path
            }
            
            if self.db_type == "chroma":
                # Get all memories
                where_clause = None
                if user_id:
                    where_clause = {"user_id": user_id}
                
                results = self.collection.get(where=where_clause)
                
                # Calculate stats
                stats["total_memories"] = len(results["ids"])
                stats["total_users"] = len(set([
                    m.get("user_id") for m in results["metadatas"] if m.get("user_id")
                ]))
                
                # Calculate size (simplified)
                stats["size_bytes"] = sum([
                    len(doc.encode("utf-8")) for doc in results["documents"]
                ])
                
                # Calculate time range
                if results["metadatas"]:
                    timestamps = [m.get("timestamp", 0) for m in results["metadatas"]]
                    stats["oldest_timestamp"] = min(timestamps)
                    stats["newest_timestamp"] = max(timestamps)
                else:
                    stats["oldest_timestamp"] = None
                    stats["newest_timestamp"] = None
            
            elif self.db_type == "faiss":
                # Filter by user_id if specified
                if user_id:
                    user_indices = [
                        i for i, uid in enumerate(self.metadata["user_ids"])
                        if uid == user_id
                    ]
                    total_memories = len(user_indices)
                else:
                    total_memories = len(self.metadata["ids"])
                
                # Calculate stats
                stats["total_memories"] = total_memories
                stats["total_users"] = len(set([
                    uid for uid in self.metadata["user_ids"] if uid
                ]))
                
                # Calculate size (simplified)
                if user_id:
                    size_bytes = sum([
                        len(self.metadata["contents"][i].encode("utf-8"))
                        for i in user_indices
                    ])
                else:
                    size_bytes = sum([
                        len(content.encode("utf-8"))
                        for content in self.metadata["contents"]
                    ])
                stats["size_bytes"] = size_bytes
                
                # Calculate time range
                if self.metadata["timestamps"]:
                    if user_id:
                        timestamps = [
                            self.metadata["timestamps"][i]
                            for i in user_indices
                        ]
                    else:
                        timestamps = self.metadata["timestamps"]
                    
                    if timestamps:
                        stats["oldest_timestamp"] = min(timestamps)
                        stats["newest_timestamp"] = max(timestamps)
                    else:
                        stats["oldest_timestamp"] = None
                        stats["newest_timestamp"] = None
                else:
                    stats["oldest_timestamp"] = None
                    stats["newest_timestamp"] = None
            
            logger.info(f"Memory stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            raise 