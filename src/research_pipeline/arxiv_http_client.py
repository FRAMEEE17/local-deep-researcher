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
        """Call ArXiv MCP server tool via direct REST endpoint."""
        if not self.session:
            raise Exception("HTTP session not initialized. Use async context manager.")
        
        # Map tool names to actual REST endpoints
        endpoint_map = {
            "search_papers": "/arxiv-mcp-server/search_papers",
            "download_paper": "/arxiv-mcp-server/download_paper", 
            "read_paper": "/arxiv-mcp-server/read_paper",
            "list_papers": "/arxiv-mcp-server/list_papers",
            "hybrid_search": "/arxiv-mcp-server/search_papers"  # hybrid_search maps to search_papers
        }
        
        endpoint = endpoint_map.get(tool_name)
        if not endpoint:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "tool": tool_name
            }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            # For list_papers, use GET with no body
            if tool_name == "list_papers":
                async with self.session.post(url) as response:
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
            else:
                print(f"DEBUG: Making request to {url}")
                print(f"DEBUG: Request body: {arguments}")
                # For other endpoints, use POST with arguments as body
                async with self.session.post(url, json=arguments) as response:
                    print(f"DEBUG: Response status: {response.status}")
                    response_text = await response.text()
                    print(f"DEBUG: Response text: {response_text[:500]}...")
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
    
    async def hybrid_search(self, query: str, max_results: int = 50,
                           search_method: str = "hybrid",
                           include_content: bool = False,
                           jina_api_key: Optional[str] = None,
                           categories: Optional[List[str]] = None) -> Dict[str, Any]:
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
    Execute ArXiv search strategy by calling the MCP server.
    
    For ArXiv papers, performs dual search:
    1. Direct paper ID search for exact paper
    2. Complex query search for related papers
    """
    import re
    start_time = time.time()
    
    async with ArxivMCPOClient(config) as client:
        try:
            # Detect ArXiv paper ID in query
            arxiv_id_pattern = r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b'
            # arxiv_match = re.search(arxiv_id_pattern, query)
            print(f"DEBUG: Checking ArXiv ID in query: {query}")
            arxiv_match = re.search(arxiv_id_pattern, query)
            print(f"DEBUG: ArXiv match found: {arxiv_match.group(1) if arxiv_match else 'None'}")
            if strategy == "arxiv_search":
                all_papers = []
                search_methods = []
                
                if arxiv_match:
                    # Dual search approach for ArXiv papers
                    paper_id = arxiv_match.group(1)
                    print(f"DEBUG: Starting dual search for paper ID: {paper_id}")
                    
                    # Strip version number for MCP server compatibility
                    clean_paper_id = paper_id.split('v')[0]  # "2410.21338v2" → "2410.21338"
                    print(f"DEBUG: Original ID: {paper_id}, Clean ID: {clean_paper_id}")
                    # Search 1: Direct paper ID search
                    print(f"DEBUG: Executing direct search with query: {paper_id}")
                    direct_result = await client.search_papers_basic(
                        query=clean_paper_id,
                        max_results=50
                    )
                    print(f"DEBUG: Direct search result: {direct_result.get('success')}")
                    if direct_result.get("success"):
                        direct_data = direct_result.get("data", {})
                        print(f"DEBUG: Direct result data type: {type(direct_data)}")
                        print(f"DEBUG: Direct result data: {direct_data}")
                        if isinstance(direct_data, str):
                            try:
                                direct_data = json.loads(direct_data)
                            except json.JSONDecodeError:
                                pass
                        
                        direct_papers = []
                        if isinstance(direct_data, list) and len(direct_data) > 0:
                            first_content = direct_data[0]
                            if hasattr(first_content, 'text'):
                                content_data = json.loads(first_content.text)
                                direct_papers = content_data.get("papers", [])
                        elif isinstance(direct_data, dict):
                            direct_papers = direct_data.get("papers", [])
                        
                        all_papers.extend(direct_papers)
                        search_methods.append(f"direct_id_search:{paper_id}")
                    print(f"DEBUG: Direct papers found: {len(direct_papers)}")
                    # Search 2: Complex query search for related papers
                    print(f"DEBUG: Executing complex search with query: {query}")
                    related_result = await client.hybrid_search(
                        query=query,
                        max_results=config.max_papers_per_search - len(all_papers),
                        search_method="arxiv_only",
                        include_content=bool(config.jina_api_key),
                        jina_api_key=config.jina_api_key
                    )
                    print(f"DEBUG: Complex search result: {related_result.get('success')}")
                    if related_result.get("success"):
                        related_data = related_result.get("data", {})
                        print(f"DEBUG: Related result data type: {type(related_data)}")
                        print(f"DEBUG: Related result data: {related_data}")
                        if isinstance(related_data, str):
                            try:
                                related_data = json.loads(related_data)
                            except json.JSONDecodeError:
                                pass
                        
                        related_papers = []
                        if isinstance(related_data, list) and len(related_data) > 0:
                            first_content = related_data[0]
                            if hasattr(first_content, 'text'):
                                content_data = json.loads(first_content.text)
                                related_papers = content_data.get("papers", [])
                        elif isinstance(related_data, dict):
                            related_papers = related_data.get("papers", [])
                        
                        # Deduplicate by paper ID
                        seen_ids = {paper.get('id') for paper in all_papers}
                        for paper in related_papers:
                            if paper.get('id') not in seen_ids:
                                all_papers.append(paper)
                                seen_ids.add(paper.get('id'))
                        
                        search_methods.append("complex_query_search")
                    
                    # Return combined results
                    return {
                        "success": True,
                        "papers": all_papers,
                        "content_extractions": [],
                        "search_methods": search_methods,
                        "execution_time": time.time() - start_time,
                        "strategy": strategy,
                        "source": "arxiv_mcp_server_dual"
                    }
                else:
                    # No ArXiv ID detected - use standard search
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