"""
HTTP client for calling ArXiv MCP server via MCPO.

This replaces direct MCP connections and redundant ArXiv functionality
in the research pipeline. All ArXiv operations are delegated to the
enhanced MCP server.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
import aiohttp
from .configuration import Configuration


class ArxivMCPOClient:
    """HTTP client for ArXiv MCP server via MCPO gateway."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.base_url = getattr(config, 'arxiv_mcp_server_url', 'http://localhost:9937')
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool via HTTP (MCPO gateway)."""
        if not self.session:
            raise Exception("HTTP session not initialized. Use async context manager.")
            
        # MCPO endpoint for MCP tool calls
        url = f"{self.base_url}/mcp/call"
        
        payload = {
            "tool": tool_name,
            "arguments": arguments
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "data": result,
                        "tool": tool_name
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "tool": tool_name
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def search_papers_basic(self, query: str, max_results: int = 10, 
                                 categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Basic ArXiv API search."""
        arguments = {
            "query": query,
            "max_results": max_results
        }
        
        if categories:
            arguments["categories"] = categories
            
        return await self._call_mcp_tool("search_papers", arguments)
    
    async def hybrid_search(self, query: str, max_results: int = 10,
                           search_method: str = "hybrid",
                           include_content: bool = False,
                           jina_api_key: Optional[str] = None,
                           categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced hybrid search (ArXiv + SearchTheArxiv + Jina AI)."""
        arguments = {
            "query": query,
            "max_results": max_results,
            "search_method": search_method,
            "include_content": include_content
        }
        
        if jina_api_key:
            arguments["jina_api_key"] = jina_api_key
            
        if categories:
            arguments["categories"] = categories
            
        return await self._call_mcp_tool("hybrid_search", arguments)
    
    async def download_paper(self, paper_id: str) -> Dict[str, Any]:
        """Download and convert paper to markdown."""
        arguments = {"paper_id": paper_id}
        return await self._call_mcp_tool("download_paper", arguments)
    
    async def read_paper(self, paper_id: str) -> Dict[str, Any]:
        """Read downloaded paper content."""
        arguments = {"paper_id": paper_id}
        return await self._call_mcp_tool("read_paper", arguments)
    
    async def list_papers(self) -> Dict[str, Any]:
        """List all downloaded papers."""
        return await self._call_mcp_tool("list_papers", {})


async def execute_arxiv_search_strategy(query: str, strategy: str, config: Configuration) -> Dict[str, Any]:
    """
    Execute ArXiv search strategy by calling the enhanced MCP server.
    
    This replaces all ArXiv functionality in search_engines.py
    """
    start_time = time.time()
    
    async with ArxivMCPOClient(config) as client:
        try:
            if strategy == "arxiv_search":
                # Academic research - use enhanced hybrid search with ArXiv focus
                result = await client.hybrid_search(
                    query=query,
                    max_results=config.max_papers_per_search,
                    search_method="arxiv_only",
                    include_content=bool(config.jina_api_key),
                    jina_api_key=config.jina_api_key
                )
                
            elif strategy == "hybrid_search":
                # Hybrid research - use full hybrid search
                result = await client.hybrid_search(
                    query=query,
                    max_results=config.max_papers_per_search,
                    search_method="hybrid",
                    include_content=bool(config.jina_api_key),
                    jina_api_key=config.jina_api_key
                )
                
            else:
                # Fallback to basic search
                result = await client.search_papers_basic(
                    query=query,
                    max_results=config.max_papers_per_search
                )
            
            execution_time = time.time() - start_time
            
            if result.get("success"):
                data = result.get("data", {})
                
                # Parse the MCP response (JSON string)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                
                # Extract papers from nested structure
                papers = []
                if isinstance(data, list) and len(data) > 0:
                    # MCP returns list of TextContent
                    first_content = data[0]
                    if hasattr(first_content, 'text'):
                        content_data = json.loads(first_content.text)
                        papers = content_data.get("papers", [])
                elif isinstance(data, dict):
                    papers = data.get("papers", [])
                
                return {
                    "success": True,
                    "papers": papers,
                    "content_extractions": data.get("content_extractions", []) if isinstance(data, dict) else [],
                    "search_methods": data.get("search_methods", []) if isinstance(data, dict) else [],
                    "execution_time": execution_time,
                    "strategy": strategy,
                    "source": "arxiv_mcp_server"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "execution_time": execution_time,
                    "strategy": strategy,
                    "source": "arxiv_mcp_server"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "strategy": strategy,
                "source": "arxiv_mcp_server"
            }


# Backward compatibility functions
async def arxiv_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    from .configuration import Configuration
    
    config = Configuration()
    result = await execute_arxiv_search_strategy(query, "arxiv_search", config)
    
    if result.get("success"):
        return result.get("papers", [])
    else:
        return []