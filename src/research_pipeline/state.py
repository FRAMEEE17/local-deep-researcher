import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import Dict, List, Any, Optional

@dataclass(kw_only=True)
class ResearchState:
    # Input/Query
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    
    # Intent Classification
    search_intent: str = field(default=None)  # academic_intent, web_intent, hybrid_intent
    intent_confidence: float = field(default=0.0)
    
    # Search Results (multiple sources)
    arxiv_results: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    web_results: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    semantic_results: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    
    # Content Extraction
    extracted_content: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    jina_extractions: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    
    # Processing & Analysis
    content_chunks: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Vector Storage & Embeddings
    vector_embeddings: Annotated[List[List[float]], operator.add] = field(default_factory=list)
    pinecone_stored: bool = field(default=False)
    vector_search_results: Annotated[List[Dict[str, Any]], operator.add] = field(default_factory=list)
    hybrid_scores: Dict[str, float] = field(default_factory=dict)
    
    # Loop Control
    research_loop_count: int = field(default=0)
    max_loops: int = field(default=3)
    
    # Output
    running_summary: str = field(default=None)
    final_summary: str = field(default=None)
    sources_gathered: Annotated[List[str], operator.add] = field(default_factory=list)
    
    # Performance Metrics
    search_strategy: str = field(default="hybrid")  # hybrid, academic_only, web_only
    processing_times: Dict[str, float] = field(default_factory=dict)
    
    query_optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    reflection_metadata: Dict[str, Any] = field(default_factory=dict)
    content_confidence_score: float = field(default=0.0)

@dataclass(kw_only=True)
class ResearchStateInput:
    """Input state for research pipeline"""
    research_topic: str = field(default=None)
    max_loops: int = field(default=3)
    search_strategy: Optional[str] = field(default=None)  # Override automatic intent routing

@dataclass(kw_only=True)
class ResearchStateOutput:
    """Output state for research pipeline"""
    final_summary: str = field(default=None)
    search_intent: str = field(default=None)
    intent_confidence: float = field(default=0.0)
    total_papers_found: int = field(default=0)
    content_extracted: int = field(default=0)
    sources_gathered: List[str] = field(default_factory=list)
    processing_times: Dict[str, float] = field(default_factory=dict)

# Backward compatibility
SummaryState = ResearchState
SummaryStateInput = ResearchStateInput  
SummaryStateOutput = ResearchStateOutput