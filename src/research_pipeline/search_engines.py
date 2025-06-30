import time
import logging
import re
import aiohttp
import asyncio
import ssl
from typing import Dict, Any, Optional, List
from research_pipeline.configuration import Configuration
from research_pipeline.arxiv_http_client import execute_arxiv_search_strategy
import requests
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from readability import Document
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False
       
logger = logging.getLogger("research_pipeline.search_engines")

def test_searxng_instances(instances):
    working = []
    for url in instances:
        try:
            timeout = 5 if 'localhost' in url or '127.0.0.1' in url else 10
            response = requests.get(f"{url}/search", params={"q": "test"}, timeout=timeout)
            if response.status_code == 200:
                working.append(url)
                print(f"✓ {url}")
            else:
                print(f"✗ {url} (status: {response.status_code})")
        except Exception as e:
            print(f"✗ {url} (error: {str(e)})")
    return working

def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate papers based on ArXiv ID."""
    seen_ids = set()
    unique_papers = []
    
    for paper in papers:
        paper_id = paper.get('id', '').replace('v1', '').replace('v2', '').replace('v3', '')
        base_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
        
        if base_id not in seen_ids:
            seen_ids.add(base_id)
            unique_papers.append(paper)
        else:
            logger.debug(f"Skipping duplicate paper: {paper.get('title', 'Unknown')}")
    
    return unique_papers

class SearchEngines:
    def __init__(self, config: Configuration):
        self.config = config
        # Content extraction settings
        self.max_content_length = getattr(config, 'max_content_length', 5000)
        self.web_timeout = getattr(config, 'web_timeout', 60)
        self.max_file_size = getattr(config, 'max_file_size', 10 * 1024 * 1024)  # 5MB
        
        # Problematic domains to skip (SSL issues, etc.)
        self.problematic_domains = [
            'kbbank.co.id', 
            'bankcentral.id',
            'malicious-site.com', 
        ]
        
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
            
            # Deduplicate papers if found
            if result.get("success") and result.get("papers"):
                result["papers"] = deduplicate_papers(result["papers"])
                logger.info(f"ArXiv search: {len(result['papers'])} unique papers after deduplication")
            
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
                "source": "searxng_enhanced",
                "web_results_count": len(results)  # Proper counting
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
                paper_id = arxiv_id_match.group(1).replace('v1', '').replace('v2', '')
                arxiv_query = f"paper {paper_id}"
                logger.info(f"Hybrid: ArXiv ID detected, using: {arxiv_query}")
            else:
                arxiv_query = query
                logger.info(f"Hybrid: Using semantic search: {arxiv_query}")
                
            arxiv_result = await execute_arxiv_search_strategy(arxiv_query, "arxiv_search", self.config)
            
            if arxiv_result.get("success"):
                arxiv_results = deduplicate_papers(arxiv_result.get("papers", []))
                logger.info(f"Hybrid: ArXiv found {len(arxiv_results)} unique papers")
            else:
                errors.append(f"ArXiv: {arxiv_result.get('error')}")
                
        except Exception as e:
            errors.append(f"ArXiv exception: {str(e)}")
        
        # Web search component with error handling
        try:
            # Clean query for web search
            web_query = query.replace("Research on", "").replace("explain", "").strip()
            if arxiv_id_match:
                web_query = f"FinTeamExperts MOE financial analysis {arxiv_id_match.group(1)}"
                
            web_results = await self._searxng_search(web_query, max_results // 2)
            logger.info(f"Hybrid: Web found {len(web_results)} results")
            
        except Exception as e:
            errors.append(f"Web exception: {str(e)}")
            logger.warning(f"Web search failed in hybrid mode: {e}")
        
        has_results = len(arxiv_results) > 0 or len(web_results) > 0
        
        return {
            "success": has_results,
            "papers": arxiv_results,
            "results": web_results,
            "strategy": strategy,
            "source": "hybrid_search_enhanced",
            "component_errors": errors if errors else None,
            "arxiv_count": len(arxiv_results),
            "web_count": len(web_results)
        }
    
    async def _searxng_search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        
        searxng_instances = [
            self.config.searxng_url,
        ]

        # Skip testing for localhost
        if "localhost" in searxng_instances[0] or "127.0.0.1" in searxng_instances[0]:
            working_instances = searxng_instances
        else:
            working_instances = test_searxng_instances(searxng_instances)
            
        if not working_instances:
            logger.warning("No working SearXNG instances found")
            return []
        
        search_params = {
            'q': query,
            'format': 'json',
            'engines': 'google,bing,duckduckgo,wikipedia',
            'categories': 'general,science',
            'safesearch': '0'
        }
        
        # Try each working instance
        for instance in working_instances:
            try:
                logger.info(f"Searching with instance: {instance}")
                
                # Increase timeout for local Docker service
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    search_url = f"{instance}/search"
                    
                    async with session.get(search_url, params=search_params) as response:
                        logger.info(f"Response status: {response.status}")
                        
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Raw response data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                            
                            results = []
                            search_results = data.get('results', [])
                            logger.info(f"Found {len(search_results)} search results")
                            
                            # Process results with better error handling
                            successful_extractions = 0
                            failed_extractions = 0
                            
                            for i, item in enumerate(search_results[:max_results]):
                                result = {
                                    'title': item.get('title', ''),
                                    'url': item.get('url', ''),
                                    'content': item.get('content', ''),
                                    'engine': item.get('engine', 'unknown'),
                                    'source': f"searxng_local_docker"
                                }
                                
                                logger.debug(f"Result {i}: {result['title'][:50]}...")
                                
                                # Optional: Fetch full page content if configured
                                if self.config.fetch_full_page and result['url']:
                                    try:
                                        content_result = await self._extract_webpage_content(result['url'], session)
                                        if content_result.get('success') and content_result.get('content'):
                                            result['full_content'] = content_result['content'][:3000]
                                            result['page_title'] = content_result.get('title', '')
                                            successful_extractions += 1
                                        else:
                                            failed_extractions += 1
                                            logger.debug(f"Content extraction failed for {result['url']}: {content_result.get('error', 'Unknown error')}")
                                    except Exception as e:
                                        failed_extractions += 1
                                        logger.debug(f"Failed to fetch full content for {result['url']}: {e}")
                                
                                results.append(result)
                            
                            logger.info(f"Returning {len(results)} processed results")
                            logger.info(f"Content extraction: {successful_extractions} successful, {failed_extractions} failed")
                            return results
                        else:
                            error_text = await response.text()
                            logger.error(f"HTTP {response.status}: {error_text}")
                            
            except Exception as e:
                logger.error(f"SearXNG instance {instance} failed: {e}")
                continue
        
        logger.warning("All SearXNG instances failed")
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
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(jina_url, headers=headers) as response:
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

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and parsed.netloc
        except Exception:
            return False
    
    def _is_html_content(self, content_type: str) -> bool:
        if not content_type:
            return True  # Assume HTML if no content type
        
        # Skip known non-HTML types
        skip_types = ['application/pdf', 'image/', 'video/', 'audio/', 
                     'application/zip', 'application/msword']
        for skip_type in skip_types:
            if skip_type in content_type:
                return False
        
        # Accept HTML-like content types
        html_types = ['text/html', 'application/xhtml', 'text/plain']
        return any(html_type in content_type for html_type in html_types)
    
    def _should_skip_domain(self, url: str) -> bool:
        """Check if domain should be skipped due to known issues."""
        return any(domain in url for domain in self.problematic_domains)
    
    def _get_encoding(self, response: aiohttp.ClientResponse, content_bytes: bytes) -> str:
        """Determine content encoding from response headers and content."""
        # Try charset from Content-Type header
        content_type = response.headers.get('content-type', '')
        charset_match = re.search(r'charset=([^;]+)', content_type)
        if charset_match:
            return charset_match.group(1).strip('"\'')
        
        # Try to detect from HTML meta tags
        try:
            preview = content_bytes[:1024].decode('ascii', errors='ignore')
            charset_match = re.search(r'<meta[^>]+charset["\s]*=["\s]*([^"\'>\s]+)', preview, re.IGNORECASE)
            if charset_match:
                return charset_match.group(1)
        except Exception:
            pass
        
        return 'utf-8'
    
    def _extract_with_beautifulsoup(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup for better parsing."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'iframe', 'noscript', 'meta', 'link']):
                element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else ''
            
            # Try to use readability if available for main content
            if HAS_READABILITY:
                try:
                    doc = Document(html)
                    content = BeautifulSoup(doc.summary(), 'html.parser').get_text(separator=' ', strip=True)
                    title = title or doc.title()
                except Exception:
                    content = self._extract_main_content(soup)
            else:
                content = self._extract_main_content(soup)
            
            # Clean up text
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'content': content,
                'title': title,
                'method': 'beautifulsoup',
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed for {url}: {e}")
            return self._extract_with_regex(html, url)
    
    def _extract_main_content(self, soup) -> str:
        """Extract main content from BeautifulSoup object."""
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '.main-content', '.post-content',
            '#content', '#main-content', '#post-content'
        ]
        
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                return ' '.join(elem.get_text(separator=' ', strip=True) for elem in elements)
        
        # Fallback: get text from body or entire document
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
        else:
            return soup.get_text(separator=' ', strip=True)
    
    def _extract_with_regex(self, html: str, url: str) -> Dict[str, Any]:
        """Fallback extraction using regex."""
        try:
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ''
            title = re.sub(r'<[^>]+>', '', title)
            
            # Remove unwanted elements
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove all HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                'content': text,
                'title': title,
                'method': 'regex',
                'word_count': len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Regex extraction failed for {url}: {e}")
            return {
                'content': '',
                'title': '',
                'method': 'failed',
                'error': str(e)
            }

    async def _extract_webpage_content(self, url: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            # Skip problematic domains
            if self._should_skip_domain(url):
                return {
                    'success': False,
                    'content': '',
                    'title': '',
                    'error': 'Skipped problematic domain'
                }
            
            # Validate URL
            if not self._is_valid_url(url):
                return {
                    'success': False,
                    'content': '',
                    'title': '',
                    'error': 'Invalid URL format'
                }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            timeout = aiohttp.ClientTimeout(total=self.web_timeout, connect=5)
            
            # Create SSL context that's more permissive for problematic certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            async with session.get(url, headers=headers, timeout=timeout, ssl=ssl_context) as response:
                if response.status != 200:
                    return {
                        'success': False,
                        'content': '',
                        'title': '',
                        'error': f'HTTP {response.status}: {response.reason}'
                    }
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not self._is_html_content(content_type):
                    return {
                        'success': False,
                        'content': '',
                        'title': '',
                        'error': f'Non-HTML content type: {content_type}'
                    }
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    return {
                        'success': False,
                        'content': '',
                        'title': '',
                        'error': f'Content too large: {content_length} bytes'
                    }
                
                # Read content with size limit
                content_bytes = b''
                async for chunk in response.content.iter_chunked(8192):
                    content_bytes += chunk
                    if len(content_bytes) > self.max_file_size:
                        logger.warning(f"Content size exceeded limit for {url}")
                        break
                
                # Decode content
                try:
                    encoding = self._get_encoding(response, content_bytes)
                    html_content = content_bytes.decode(encoding, errors='replace')
                except (UnicodeDecodeError, LookupError) as e:
                    logger.warning(f"Encoding error for {url}: {e}")
                    html_content = content_bytes.decode('utf-8', errors='replace')
                
                # Extract content using best available method
                if HAS_BS4:
                    result = self._extract_with_beautifulsoup(html_content, url)
                else:
                    result = self._extract_with_regex(html_content, url)
                
                # Truncate if needed
                if len(result['content']) > self.max_content_length:
                    result['content'] = result['content'][:self.max_content_length] + '...'
                    result['truncated'] = True
                
                result['success'] = True
                return result
                
        except aiohttp.ClientConnectorSSLError as e:
            logger.warning(f"SSL error for {url}: {e}")
            return {
                'success': False,
                'content': '',
                'title': '',
                'error': f'SSL certificate error: {str(e)}'
            }
        except aiohttp.ClientConnectorCertificateError as e:
            logger.warning(f"Certificate error for {url}: {e}")
            return {
                'success': False,
                'content': '',
                'title': '',
                'error': f'Certificate error: {str(e)}'
            }
        except asyncio.TimeoutError:
            logger.warning(f"Timeout extracting content from {url}")
            return {
                'success': False,
                'content': '',
                'title': '',
                'error': 'Request timeout'
            }
        except aiohttp.ClientError as e:
            logger.warning(f"Client error extracting content from {url}: {e}")
            return {
                'success': False,
                'content': '',
                'title': '',
                'error': f'Network error: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error extracting content from {url}: {e}")
            return {
                'success': False,
                'content': '',
                'title': '',
                'error': f'Extraction error: {str(e)}'
            }


def create_search_engines(config: Optional[Configuration] = None) -> SearchEngines:
    if config is None:
        config = Configuration()
    
    logger.info("Creating SearchEngines instance")
    return SearchEngines(config)