"""
Direct client for ArXiv MCP server tools.

This bypasses MCP protocol issues and calls the ArXiv tools directly.
All ArXiv operations (ArXiv API + SearchTheArxiv + Jina AI) are handled here.
"""

import sys
import asyncio
import time
from typing import Dict, List, Any, Optional
import logging 
import re
logger = logging.getLogger("research_pipeline.arxiv_client")
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
    """Direct client for ArXiv server tools."""
    
    def __init__(self, config: Configuration):
        self.config = config
        
    async def search_papers_basic(self, query: str, max_results: int = 10, 
                                 categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Basic ArXiv API search using  server."""
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
    Execute ArXiv search strategy by calling the MCP server.
    with comprehensive logging for flow verification.
    
    For ArXiv papers, performs dual search:
    1. Direct paper ID search for exact paper
    2. Complex query search for related papers
    """
    logger.info("=" * 70)
    logger.info("EXECUTING: execute_arxiv_search_strategy()")
    logger.info("=" * 70)
    logger.info(f"INPUT: query = '{query}'")
    logger.info(f"INPUT: strategy = '{strategy}'")
    logger.info(f"INPUT: config type = {config.__class__.__name__}")
    
    start_time = time.time()
    
    # ArXiv ID detection with logging
    arxiv_id_pattern = r'\b(\d{4}\.\d{4,5}(?:v\d+)?)'
    arxiv_match = re.search(arxiv_id_pattern, query)
    
    if arxiv_match:
        paper_id = arxiv_match.group(1)
        logger.info(f"🔍 ArXiv ID detected in query: {paper_id}")
        logger.info(f"📋 Search mode: Direct paper ID lookup + related papers")
    else:
        logger.info(f"🔍 No ArXiv ID detected - using semantic search")
        logger.info(f"📋 Search mode: Semantic/keyword search only")
    
    try:
        # Initialize MCP client with logging
        logger.info("🔌 Initializing ArXiv MCP client...")
        from research_pipeline.arxiv_http_client import ArxivMCPOClient
        async with ArxivMCPOClient(config) as client:
            client_init_time = time.time() - start_time
            logger.info(f"✅ ArXiv MCP client initialized in {client_init_time:.2f}s")
            
            # Determine search strategy based on query analysis
            if arxiv_match:
                paper_id = arxiv_match.group(1)
                logger.info(f"📄 Executing direct paper search for ID: {paper_id}")
                
                # Try direct paper search first
                try:
                    direct_search_start = time.time()
                    result = await client.search_papers(f"id:{paper_id}", max_results=1)
                    direct_search_time = time.time() - direct_search_start
                    
                    logger.info(f"⏱️  Direct paper search completed in {direct_search_time:.2f}s")
                    
                    if result and len(result) > 0:
                        logger.info(f"✅ Direct paper found: {result[0].get('title', 'Unknown title')[:60]}...")
                        
                        # Get related papers
                        if len(query.split()) > 1:  # Has additional keywords
                            logger.info("🔗 Searching for related papers...")
                            related_start = time.time()
                            
                            # Extract additional keywords from query
                            clean_query = re.sub(arxiv_id_pattern, '', query).strip()
                            if clean_query:
                                related_results = await client.search_papers(clean_query, max_results=5)
                                related_time = time.time() - related_start
                                
                                logger.info(f"⏱️  Related papers search completed in {related_time:.2f}s")
                                logger.info(f"📊 Related papers found: {len(related_results) if related_results else 0}")
                                
                                # Combine results
                                if related_results:
                                    result.extend(related_results)
                                    logger.info(f"📋 Total papers after combining: {len(result)}")
                            else:
                                logger.info("📋 No additional keywords for related search")
                    else:
                        logger.warning(f"⚠️  Direct paper search returned no results for ID: {paper_id}")
                        
                        # Fallback to semantic search
                        logger.info("🔄 Falling back to semantic search...")
                        fallback_start = time.time()
                        result = await client.search_papers(query, max_results=10)
                        fallback_time = time.time() - fallback_start
                        
                        logger.info(f"⏱️  Fallback search completed in {fallback_time:.2f}s")
                        logger.info(f"📊 Fallback results: {len(result) if result else 0}")
                        
                except Exception as e:
                    logger.error(f"❌ Direct paper search failed: {str(e)}")
                    logger.info("🔄 Falling back to semantic search...")
                    
                    fallback_start = time.time()
                    result = await client.search_papers(query, max_results=10)
                    fallback_time = time.time() - fallback_start
                    
                    logger.info(f"⏱️  Fallback search completed in {fallback_time:.2f}s")
                    logger.info(f"📊 Fallback results: {len(result) if result else 0}")
            else:
                # Standard semantic search
                logger.info("🔍 Executing semantic search...")
                semantic_start = time.time()
                result = await client.search_papers(query, max_results=10)
                semantic_time = time.time() - semantic_start
                
                logger.info(f"⏱️  Semantic search completed in {semantic_time:.2f}s")
                logger.info(f"📊 Semantic search results: {len(result) if result else 0}")
            
            # Process and validate results
            execution_time = time.time() - start_time
            
            if result and isinstance(result, list):
                papers_count = len(result)
                logger.info("=" * 50)
                logger.info("✅ SEARCH RESULTS SUMMARY")
                logger.info("=" * 50)
                logger.info(f"📊 Total papers found: {papers_count}")
                logger.info(f"⏱️  Total execution time: {execution_time:.2f}s")
                
                # Log details of first few papers
                for i, paper in enumerate(result[:3]):
                    paper_id = paper.get('id', 'Unknown ID')
                    title = paper.get('title', 'Unknown title')
                    authors = paper.get('authors', [])
                    
                    logger.info(f"📄 Paper {i+1}:")
                    logger.info(f"    ID: {paper_id}")
                    logger.info(f"    Title: {title[:80]}...")
                    logger.info(f"    Authors: {len(authors)} author(s)")
                
                if papers_count > 3:
                    logger.info(f"    ... and {papers_count - 3} more papers")
                
                # Prepare success response
                response = {
                    "success": True,
                    "papers": result,
                    "execution_time": execution_time,
                    "strategy": strategy,
                    "source": "arxiv_mcp_server",
                    "search_metadata": {
                        "has_arxiv_id": arxiv_match is not None,
                        "paper_id": arxiv_match.group(1) if arxiv_match else None,
                        "search_type": "direct+related" if arxiv_match else "semantic",
                        "papers_found": papers_count
                    }
                }
                
                logger.info("✅ execute_arxiv_search_strategy() SUCCESS")
                logger.info("=" * 70)
                
                return response
                
            else:
                logger.warning("⚠️  Search returned no valid results")
                logger.info(f"📋 Result type: {type(result)}")
                logger.info(f"📋 Result content: {str(result)[:100]}...")
                
                response = {
                    "success": False,
                    "error": "No papers found or invalid result format",
                    "papers": [],
                    "execution_time": execution_time,
                    "strategy": strategy,
                    "source": "arxiv_mcp_server"
                }
                
                logger.warning("⚠️  execute_arxiv_search_strategy() NO RESULTS")
                logger.info("=" * 70)
                
                return response
                
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error("=" * 50)
        logger.error("❌ SEARCH EXECUTION FAILED")
        logger.error("=" * 50)
        logger.error(f"Exception: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Execution time before failure: {execution_time:.2f}s")
        
        # Log additional context for debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        response = {
            "success": False,
            "error": str(e),
            "papers": [],
            "execution_time": execution_time,
            "strategy": strategy,
            "source": "arxiv_mcp_server",
            "error_context": {
                "exception_type": type(e).__name__,
                "has_arxiv_id": arxiv_match is not None if 'arxiv_match' in locals() else None
            }
        }
        
        logger.error("❌ execute_arxiv_search_strategy() FAILED")
        logger.info("=" * 70)
        
        return response


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