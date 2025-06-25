import asyncio
import time
import requests
from typing import Dict, Any
from langchain_community.utilities import SearxSearchWrapper
from .configuration import Configuration
from .arxiv_http_client import execute_arxiv_search_strategy

class SearchEngines:
    """Search engines that work with MCPO (no direct MCP connections)"""
    
    def __init__(self, config: Configuration):
        self.config = config
        
        # SearXNG wrapper for web search only
        self.searxng = SearxSearchWrapper(searx_host=config.searxng_url)
        
        # Request timeout setting
        self.request_timeout = getattr(config, 'request_timeout', 30)
    
    # SearchTheArxiv and Jina AI functionality moved to ArXiv MCP server
    # All ArXiv-related operations are now handled by arxiv_http_client.py
    
    def search_searxng(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search using SearXNG (web search)"""
        try:
            results = self.searxng.run(query)
            
            web_results = []
            if results:
                lines = results.split('\n')
                for line in lines[:max_results]:
                    if line.strip():
                        web_results.append({
                            'title': line.strip(),
                            'content': line.strip(),
                            'url': '',  
                            'search_source': 'searxng'
                        })
            
            return {
                'success': True,
                'results': web_results,
                'count': len(web_results),
                'method': 'searxng'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'searxng',
                'results': []
            }
    
    # Jina AI content extraction moved to ArXiv MCP server
    # Use arxiv_http_client.py for content extraction capabilities
    
    # All hybrid search, vector storage, and content extraction 
    # functionality moved to ArXiv MCP server
    # Use arxiv_http_client.py to access these capabilities
    
    async def execute_search_strategy(self, query: str, strategy: str, max_results: int = 10) -> Dict[str, Any]:
        """Execute search based on intent routing strategy"""
        start_time = time.time()
        
        if strategy in ["arxiv_search", "hybrid_search"]:
            # ArXiv and Hybrid searches → Call enhanced ArXiv MCP server
            result = await execute_arxiv_search_strategy(query, strategy, self.config)
            
        elif strategy == "web_search":
            # Web Search Query → SearXNG only
            result = await asyncio.to_thread(self.search_searxng, query, max_results)
            
        else:
            # Default fallback to web search
            result = await asyncio.to_thread(self.search_searxng, query, max_results)
        
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        result['strategy'] = strategy
        
        return result

# Note: ArXiv MCP operations should be handled by MCPO context
# This module provides complementary search capabilities
# The actual ArXiv MCP calls should be made through OpenWebUI's MCPO interface

def create_search_engines(config: Configuration) -> SearchEngines:
    """Factory function to create search engines"""
    return SearchEngines(config)