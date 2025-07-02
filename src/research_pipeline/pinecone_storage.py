import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

PINECONE_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import sys
    # Check if pinecone-client is causing conflicts
    if 'pinecone' in sys.modules:
        del sys.modules['pinecone']
    
    # Try importing Pinecone and required components
    import pinecone as pc_module
    if hasattr(pc_module, 'Pinecone'):
        Pinecone = pc_module.Pinecone
        ServerlessSpec = pc_module.ServerlessSpec
        PINECONE_AVAILABLE = True
    else:
        print("Pinecone package installed but Pinecone class not found")
except Exception as e:
    print(f"Pinecone not available: {e}")
    # Create dummy classes for fallback
    class Pinecone:
        pass
    class ServerlessSpec:
        pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PaperMetadata:
    """Structure for paper metadata in Pinecone"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str
    similarity_score: float = 0.0


class PineconeArxivStorage:
    """Manages ArXiv paper storage and retrieval in Pinecone"""
    
    def __init__(self, api_key: str, index_name: str = "arxiv-metadata"):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone package not available. Install with: pip install pinecone-client")
        
        self.api_key = api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        self._setup_index()
    
    def _setup_index(self):
        """Initialize or connect to Pinecone index"""
        existing_indexes = self.pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if self.index_name not in index_names:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # XLM-RoBERTa base embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
    
    def create_paper_id(self, arxiv_id: str) -> str:
        """Create consistent paper ID for Pinecone storage"""
        return hashlib.md5(arxiv_id.encode()).hexdigest()
    
    def store_papers(self, papers: List[Dict[str, Any]], embeddings: List[List[float]]) -> Dict[str, Any]:
        """Store papers with embeddings in Pinecone"""
        if not self.index:
            return {"success": False, "error": "Index not initialized"}
        
        vectors = []
        for paper, embedding in zip(papers, embeddings):
            paper_id = self.create_paper_id(paper['id'])
            
            metadata = {
                "arxiv_id": paper['id'],
                "title": paper.get('title', ''),
                "authors": paper.get('authors', [])[:5],  # Pinecone metadata limits
                "abstract": paper.get('summary', '')[:1000],  # Truncate for storage
                "categories": paper.get('categories', [])[:5],
                "published": paper.get('published', ''),
                "pdf_url": paper.get('pdf_url', ''),
                "updated": paper.get('updated', ''),
            }
            
            vectors.append({
                "id": paper_id,
                "values": embedding,
                "metadata": metadata
            })
        
        try:
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            return {
                "success": True,
                "stored_count": len(vectors),
                "index_stats": self.index.describe_index_stats()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_papers(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[PaperMetadata]:
        """Search papers using vector similarity"""
        if not self.index:
            return []
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            papers = []
            for match in results.matches:
                metadata = match.metadata
                paper = PaperMetadata(
                    arxiv_id=metadata.get('arxiv_id', ''),
                    title=metadata.get('title', ''),
                    authors=metadata.get('authors', []),
                    abstract=metadata.get('abstract', ''),
                    categories=metadata.get('categories', []),
                    published=metadata.get('published', ''),
                    pdf_url=metadata.get('pdf_url', ''),
                    similarity_score=match.score
                )
                papers.append(paper)
            
            return papers
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        categories: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None,
        min_similarity: float = 0.7,
        top_k: int = 10
    ) -> List[PaperMetadata]:
        """Perform hybrid search with semantic + metadata filtering"""
        filter_conditions = {}
        
        # Category filtering
        if categories:
            filter_conditions["categories"] = {"$in": categories}
        
        # Date range filtering (if supported by metadata format)
        if date_range:
            if date_range.get('start'):
                filter_conditions["published"] = {"$gte": date_range['start']}
            if date_range.get('end'):
                if "published" in filter_conditions:
                    filter_conditions["published"]["$lte"] = date_range['end']
                else:
                    filter_conditions["published"] = {"$lte": date_range['end']}
        
        # Search with filters
        papers = self.search_papers(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results to filter by similarity
            filter_dict=filter_conditions if filter_conditions else None
        )
        
        # Filter by minimum similarity
        filtered_papers = [
            paper for paper in papers 
            if paper.similarity_score >= min_similarity
        ]
        
        return filtered_papers[:top_k]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_papers(self, arxiv_ids: List[str]) -> Dict[str, Any]:
        """Delete papers from index"""
        if not self.index:
            return {"success": False, "error": "Index not initialized"}
        
        try:
            paper_ids = [self.create_paper_id(arxiv_id) for arxiv_id in arxiv_ids]
            self.index.delete(ids=paper_ids)
            return {"success": True, "deleted_count": len(paper_ids)}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PineconeStorageManager:
    """High-level manager for Pinecone storage operations"""
    
    def __init__(self, config):
        self.config = config
        self.storage = None
        
        if config.pinecone_api_key and PINECONE_AVAILABLE:
            try:
                self.storage = PineconeArxivStorage(
                    api_key=config.pinecone_api_key,
                    index_name=config.pinecone_index_name
                )
            except Exception as e:
                print(f"Failed to initialize Pinecone storage: {e}")
    
    def is_available(self) -> bool:
        """Check if Pinecone storage is available"""
        return self.storage is not None
    
    def store_search_results(self, papers: List[Dict], embeddings: List[List[float]]) -> Dict[str, Any]:
        """Store search results with embeddings"""
        if not self.is_available():
            return {"success": False, "error": "Pinecone storage not available"}
        
        return self.storage.store_papers(papers, embeddings)
    
    def search_similar_papers(
        self,
        query_embedding: List[float],
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[PaperMetadata]:
        """Search for similar papers"""
        if not self.is_available():
            return []
        
        return self.storage.search_papers(query_embedding, top_k, filters)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        if not self.is_available():
            return {"error": "Pinecone storage not available"}
        
        return self.storage.get_index_stats()