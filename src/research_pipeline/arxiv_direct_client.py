"""
Direct client for ArXiv MCP server tools.

This bypasses MCP protocol issues and calls the enhanced ArXiv tools directly.
All ArXiv operations (ArXiv API + SearchTheArxiv + Jina AI) are handled here.
"""

import sys
import asyncio
import time
from typing import Dict, List, Any, Optional

# Add server path for direct tool access
sys.path.insert(0, '/home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server/src')

try:
    from arxiv_mcp_server.tools import handle_search, handle_hybrid_search, handle_download, handle_read_paper
    ARXIV_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"ArXiv tools not available: {e}")
    ARXIV_TOOLS_AVAILABLE = False

from .configuration import Configuration


class ArxivDirectClient:
    """Direct client for enhanced ArXiv server tools."""
    
    def __init__(self, config: Configuration):
        self.config = config
        
    async def search_papers_basic(self, query: str, max_results: int = 10, 
                                 categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Basic ArXiv API search using enhanced server."""
        if not ARXIV_TOOLS_AVAILABLE:
            return {"success": False, "error": "ArXiv tools not available"}
            
        try:
            arguments = {
                "query": query,
                "max_results": max_results
            }
            
            if categories:
                arguments["categories"] = categories
                
            # Call the server tool directly
            result = await handle_search(arguments)
            
            if result and len(result) > 0:
                # Parse TextContent response
                content_text = result[0].text
                import json
                data = json.loads(content_text)
                
                return {
                    "success": True,
                    "papers": data.get("papers", []),
                    "total_results": data.get("total_results", 0),
                    "source": "arxiv_api_direct"
                }
            else:
                return {"success": False, "error": "No results returned"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def hybrid_search(self, query: str, max_results: int = 10,
                           search_method: str = "hybrid",
                           include_content: bool = False,
                           jina_api_key: Optional[str] = None,
                           categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced hybrid search (ArXiv + SearchTheArxiv + Jina AI)."""
        if not ARXIV_TOOLS_AVAILABLE:
            return {"success": False, "error": "ArXiv tools not available"}
            
        try:
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
                
            # Call the enhanced hybrid search tool directly
            result = await handle_hybrid_search(arguments)
            
            if result and len(result) > 0:
                # Parse TextContent response
                content_text = result[0].text
                import json
                data = json.loads(content_text)
                
                return {
                    "success": True,
                    "papers": data.get("papers", []),
                    "content_extractions": data.get("content_extractions", []),
                    "search_methods": data.get("search_methods", []),
                    "total_papers": data.get("total_papers", 0),
                    "source": "hybrid_search_direct"
                }
            else:
                return {"success": False, "error": "No results returned"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def download_paper(self, paper_id: str) -> Dict[str, Any]:
        """Download and convert paper to markdown."""
        if not ARXIV_TOOLS_AVAILABLE:
            return {"success": False, "error": "ArXiv tools not available"}
            
        try:
            arguments = {"paper_id": paper_id}
            result = await handle_download(arguments)
            
            if result and len(result) > 0:
                content_text = result[0].text
                import json
                data = json.loads(content_text)
                return {"success": True, "data": data}
            else:
                return {"success": False, "error": "No results returned"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def read_paper(self, paper_id: str) -> Dict[str, Any]:
        """Read downloaded paper content."""
        if not ARXIV_TOOLS_AVAILABLE:
            return {"success": False, "error": "ArXiv tools not available"}
            
        try:
            arguments = {"paper_id": paper_id}
            result = await handle_read_paper(arguments)
            
            if result and len(result) > 0:
                content_text = result[0].text
                import json
                data = json.loads(content_text)
                return {"success": True, "data": data}
            else:
                return {"success": False, "error": "No results returned"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


async def execute_arxiv_search_strategy(query: str, strategy: str, config: Configuration) -> Dict[str, Any]:
    """
    Execute ArXiv search strategy using the enhanced direct client.
    
    This replaces all ArXiv functionality in the research pipeline.
    """
    start_time = time.time()
    
    client = ArxivDirectClient(config)
    
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
            return {
                "success": True,
                "papers": result.get("papers", []),
                "content_extractions": result.get("content_extractions", []),
                "search_methods": result.get("search_methods", []),
                "execution_time": execution_time,
                "strategy": strategy,
                "source": "arxiv_direct_server"
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "execution_time": execution_time,
                "strategy": strategy,
                "source": "arxiv_direct_server"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "strategy": strategy,
            "source": "arxiv_direct_server"
        }


# Backward compatibility
async def arxiv_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    from .configuration import Configuration
    
    config = Configuration()
    result = await execute_arxiv_search_strategy(query, "arxiv_search", config)
    
    if result.get("success"):
        return result.get("papers", [])
    else:
        return []