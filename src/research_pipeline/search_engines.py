import time
import logging
import re
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from research_pipeline.configuration import Configuration
from research_pipeline.arxiv_http_client import execute_arxiv_search_strategy
import requests
logger = logging.getLogger("research_pipeline.search_engines")

def test_searxng_instances(instances):
    working = []
    for url in instances:
        try:
            response = requests.get(f"{url}/search", params={"q": "test"}, timeout=10)
            if response.status_code == 200:
                working.append(url)
                print(f"✓ {url}")
            else:
                print(f"✗ {url} (status: {response.status_code})")
        except Exception as e:
            print(f"✗ {url} (error: {str(e)})")
    return working
class SearchEngines:
    def __init__(self, config: Configuration):
        self.config = config
        logger.info(f"SearchEngines initialized with config")
    
    async def execute_search_strategy(self, query: str, strategy: str, max_results: int = 10) -> Dict[str, Any]:
        logger.info(f"SearchEngines routing {strategy} for query: '{query}'")
        
        start_time = time.time()
        
        try:
            if strategy == "arxiv_search":
                result = await execute_arxiv_search_strategy(query, strategy, self.config)
                
            elif strategy == "web_search":
                result = await self._execute_web_search_strategy(query, strategy, max_results)
                
            elif strategy == "hybrid_search":
                result = await self._execute_hybrid_search_strategy(query, strategy, max_results)
                
            else:
                logger.warning(f"Unknown strategy: {strategy}, falling back to hybrid")
                result = await self._execute_hybrid_search_strategy(query, strategy, max_results)
            
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result["search_execution_time"] = execution_time
                result["executed_strategy"] = strategy
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Search strategy failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "strategy": strategy,
                "search_execution_time": execution_time
            }
    
    async def _execute_arxiv_search_strategy(self, query: str, strategy: str, max_results: int) -> Dict[str, Any]:
        """Execute ArXiv search using MCP server."""
        try:
            result = await execute_arxiv_search_strategy(query, strategy, self.config)
            return result
        except Exception as e:
            return {"success": False, "error": f"ArXiv search failed: {str(e)}", "strategy": strategy}
    
    async def _execute_web_search_strategy(self, query: str, strategy: str, max_results: int) -> Dict[str, Any]:
        try:
            results = await self._searxng_search(query, max_results)
            return {
                "success": True,
                "results": results,
                "strategy": strategy,
                "source": "searxng_enhanced"
            }
        except Exception as e:
            return {"success": False, "error": f"Web search failed: {str(e)}", "strategy": strategy}
    
    async def _execute_hybrid_search_strategy(self, query: str, strategy: str, max_results: int) -> Dict[str, Any]:
        logger.info("Executing hybrid search strategy")
        
        # Extract ArXiv ID for direct search
        arxiv_id_match = re.search(r'\b(\d{4}\.\d{4,5}(?:v\d+)?)', query)
        
        arxiv_results = []
        web_results = []
        errors = []
        
        # ArXiv search component
        try:
            if arxiv_id_match:
                # paper_id = arxiv_id_match.group(1)
                paper_id = arxiv_id_match.group(1).replace('v1', '').replace('v2', '')
                arxiv_query = f"paper {paper_id}"
                logger.info(f"Hybrid: ArXiv ID detected, using: {arxiv_query}")
            else:
                arxiv_query = query
                logger.info(f"Hybrid: Using semantic search: {arxiv_query}")
                
            arxiv_result = await execute_arxiv_search_strategy(arxiv_query, "arxiv_search", self.config)
            
            if arxiv_result.get("success"):
                arxiv_results = arxiv_result.get("papers", [])
                logger.info(f"Hybrid: ArXiv found {len(arxiv_results)} papers")
            else:
                errors.append(f"ArXiv: {arxiv_result.get('error')}")
                
        except Exception as e:
            errors.append(f"ArXiv exception: {str(e)}")
        
        # Web search component
        try:
            # Clean query for web search
            web_query = query.replace("Research on", "").replace("explain", "").strip()
            if arxiv_id_match:
                web_query = f"FinTeamExperts MOE financial analysis {arxiv_id_match.group(1)}"
                
            web_results = await self._searxng_search(web_query, max_results // 2)
            logger.info(f"Hybrid: Web found {len(web_results)} results")
            
        except Exception as e:
            errors.append(f"Web exception: {str(e)}")
        
        has_results = len(arxiv_results) > 0 or len(web_results) > 0
        
        return {
            "success": has_results,
            "papers": arxiv_results,
            "results": web_results,
            "strategy": strategy,
            "source": "hybrid_search_enhanced",
            "component_errors": errors if errors else None
        }
    
    async def _searxng_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        
        searxng_instances = [
            "https://search.inetol.net/",
        ]


        working_instances = test_searxng_instances(searxng_instances)
        
        search_params = {
            'q': query,
            'format': 'json',
            'engines': 'google,bing,duckduckgo,wikipedia',
            'categories': 'general,science',
            'safesearch': '0'
        }
        
        for instance in searxng_instances:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    search_url = f"{instance}/search"
                    
                    async with session.get(search_url, params=search_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            results = []
                            for item in data.get('results', [])[:max_results]:
                                result = {
                                    'title': item.get('title', ''),
                                    'url': item.get('url', ''),
                                    'content': item.get('content', ''),
                                    'engine': item.get('engine', 'unknown'),
                                    'source': f"searxng_{instance.split('//')[1].split('.')[0]}"
                                }
                                
                    
                                if self.config.fetch_full_page and result['url']:
                                    try:
                                        content = await self._extract_webpage_content(result['url'], session)
                                        if content:
                                            result['full_content'] = content[:3000]
                                    except:
                                        pass
                                
                                results.append(result)
                            
                            return results
                            
            except Exception as e:
                logger.warning(f"SearXNG instance {instance} failed: {e}")
                continue
        
        return []
    
    async def extract_content_jina(self, url: str) -> Dict[str, Any]:
        """Extract content from URL using Jina AI reader service."""
        if not self.config.jina_api_key:
            return {
                "success": False,
                "error": "Jina API key not configured",
                "content": "",
                "url": url
            }
        
        try:
            # Use Jina AI reader service
            jina_url = f"https://r.jina.ai/{url}"
            headers = {"Authorization": f"Bearer {self.config.jina_api_key}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(jina_url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        return {
                            "success": True,
                            "content": content,
                            "url": url,
                            "method": "jina_ai"
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "content": "",
                            "url": url
                        }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": "",
                "url": url
            }

    async def _extract_webpage_content(self, url: str, session: aiohttp.ClientSession) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
            
            async with session.get(url, headers=headers, timeout=8) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Basic HTML cleaning
                    html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', html)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    return text[:5000]
        except:
            pass
        return ""


def create_search_engines(config: Optional[Configuration] = None) -> SearchEngines:
    if config is None:
        config = Configuration()
    
    logger.info("Creating SearchEngines instance")
    return SearchEngines(config)