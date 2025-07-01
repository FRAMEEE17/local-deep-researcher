import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig

class SearchAPI(Enum):
    SEARXNG = "searxng"  # Primary web search
    ARXIV_MCP = "arxiv_mcp"  # ArXiv MCP server
    SEARCHTHEARXIV = "searchthearxiv"  # Semantic ArXiv search
    HYBRID = "hybrid"  # Combined approach
    JINA = "jina"  # Content extraction

class Configuration(BaseModel):
    max_research_loops: int = Field(
        default=3,
        title="Max Research Loops",
        description="Maximum number of research loops for reflection"
    )
    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform"
    )
    local_llm: str = Field(
        default="deepseek-r1:70b",
        title="LLM Model Name",
        description="Name of the LLM model to use (provider-specific)"
    )
    # Provider-specific models
    ollama_model: str = Field(
        default="deepseek-r1:70b",
        title="Ollama Model",
        description="Model name for Ollama provider"
    )
    lmstudio_model: str = Field(
        default="deepseek-r1:70b", 
        title="LMStudio Model",
        description="Model name for LMStudio provider"
    )
    nvidia_nim_model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        title="NVIDIA NIM Model", 
        description="Model name for NVIDIA NIM provider"
    )
    llm_provider: Literal["ollama", "lmstudio", "nvidia_nim"] = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider for the LLM (Ollama, LMStudio, or NVIDIA NIM)"
    )
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng", "arxiv", "jina", "hybrid"] = Field(
        default="searxng",
        title="Search API",
        description="Search API to use (web search or academic papers)"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API"
    )
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        title="NVIDIA NIM Base URL",
        description="Base URL for NVIDIA NIM API"
    )
    nvidia_api_key: Optional[str] = Field(
        default=None,
        title="NVIDIA API Key",
        description="API key for NVIDIA NIM (or set NVIDIA_API_KEY env var)"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )

    # Intent Classification 
    intent_model_path: str = Field(
        default="/home/siamai/deepsad/local-deep-researcher/data/intent_classifier_xlm.pth",
        title="Intent Model Path",
        description="Path to trained XLM-RoBERTa intent classification model"
    )
    tokenizer_path: str = Field(
        default="/home/siamai/deepsad/local-deep-researcher/data/tokenizer_final",
        title="Tokenizer Path", 
        description="Path to XLM-RoBERTa tokenizer"
    )
    label_encoder_path: str = Field(
        default="/home/siamai/deepsad/local-deep-researcher/data/label_encoder_xlm.pkl",
        title="Label Encoder Path",
        description="Path to label encoder for intent classification"
    )
    
    # Search Configuration
    searxng_url: str = Field(
        default="http://localhost:8001",
        title="SearXNG URL",
        description="URL for SearXNG search engine"
    )
    arxiv_mcp_server_path: str = Field(
        default="/home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server",
        title="ArXiv MCP Server Path",
        description="Path to ArXiv MCP server directory"
    )
    jina_api_key: Optional[str] = Field(
        default=None,
        title="Jina AI API Key",
        description="API key for Jina AI content extraction"
    )

    
    # Content extraction settings
    max_content_length: int = Field(
        default=5000,
        title="Max Content Length",
        description="Maximum characters to extract from each page"
    )
    web_timeout: int = Field(
        default=60,
        title="Web Request Timeout",
        description="Timeout for web requests in seconds"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,
        title="Max File Size",
        description="Maximum file size to download (10MB default)"
    )
    
    # Processing Limits
    max_papers_per_search: int = Field(
        default=10,
        title="Max Papers Per Search",
        description="Maximum papers to retrieve per search"
    )
    max_content_extractions: int = Field(
        default=3,
        title="Max Content Extractions", 
        description="Maximum papers to extract full content from"
    )
    # Performance Settings
    enable_async: bool = Field(
        default=True,
        title="Enable Async Processing",
        description="Enable asynchronous concurrent operations"
    )
    max_concurrent_requests: int = Field(
        default=5,
        title="Max Concurrent Requests",
        description="Maximum concurrent HTTP requests"
    )
    
    # OpenWebUI/MCPO Integration
    mcpo_enabled: bool = Field(
        default=True,
        title="Enable MCPO",
        description="Enable Model Context Protocol for OpenWebUI"
    )
    arxiv_mcp_server_url: str = Field(
        default="http://localhost:9937",
        title="ArXiv MCP Server URL",
        description="URL for ArXiv MCP server via MCPO gateway"
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        
        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        
        return cls(**values)
    
    def get_model_name(self) -> str:
        """Get the appropriate model name for the current provider."""
        if self.llm_provider == "nvidia_nim":
            return self.nvidia_nim_model
        elif self.llm_provider == "lmstudio":
            return self.lmstudio_model
        elif self.llm_provider == "ollama":
            return self.ollama_model
        else:
            # Fallback to local_llm for backward compatibility
            return self.local_llm