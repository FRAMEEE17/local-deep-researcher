import arxiv
import json
import requests
import asyncio
import urllib.parse
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dateutil import parser
import mcp.types as types
from ..config import Settings

settings = Settings()

hybrid_search_tool = types.Tool(
    name="hybrid_search",
    description="Hybrid search using ArXiv API, SearchTheArxiv metadata, and Jina AI content extraction",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 10},
            "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "include_content": {"type": "boolean", "default": False, "description": "Extract full content with Jina AI"},
            "jina_api_key": {"type": "string", "description": "Jina AI API key for content extraction"},
            "search_method": {"type": "string", "enum": ["arxiv_only", "searchthearxiv_only", "hybrid"], "default": "hybrid"}
        },
        "required": ["query"],
    },
)


async def search_arxiv_api(query: str, max_results: int = 50, 
                          date_from: Optional[str] = None, 
                          date_to: Optional[str] = None,
                          categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Search using ArXiv API."""
    try:
        client = arxiv.Client()
        
        # Build search query with category filtering
        search_query = query
        
        # Add field specifier if not already present
        if not any(field in search_query for field in ["all:", "ti:", "abs:", "au:", "cat:"]):
            if '"' in search_query:
                search_query = f"all:{search_query}"
            else:
                terms = search_query.split()
                if len(terms) > 1:
                    search_query = " AND ".join(f"all:{term}" for term in terms)
                else:
                    search_query = f"all:{search_query}"

        if categories:
            category_filter = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"({search_query}) AND ({category_filter})"

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # Process results with date filtering
        results = []
        try:
            date_from_parsed = (
                parser.parse(date_from).replace(tzinfo=timezone.utc)
                if date_from else None
            )
            date_to_parsed = (
                parser.parse(date_to).replace(tzinfo=timezone.utc)
                if date_to else None
            )
        except (ValueError, TypeError):
            date_from_parsed = date_to_parsed = None

        for paper in client.results(search):
            # Date filtering
            if date_from_parsed and paper.published < date_from_parsed:
                continue
            if date_to_parsed and paper.published > date_to_parsed:
                continue
                
            results.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "categories": paper.categories,
                "published": paper.published.isoformat(),
                "url": paper.pdf_url,
                "arxiv_url": f"https://arxiv.org/abs/{paper.get_short_id()}",
                "pdf_url": paper.pdf_url,
                "source": "arxiv_api",
                "resource_uri": f"arxiv://{paper.get_short_id()}",
            })

            if len(results) >= max_results:
                break

        return results

    except Exception as e:
        raise Exception(f"ArXiv API search failed: {str(e)}")


async def search_searchthearxiv_api(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Search using SearchTheArxiv API for semantic search."""
    try:
        url = "https://searchthearxiv.com/search"
        encoded_query = urllib.parse.quote(query)
        params = {"query": encoded_query}
        
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-requested-with": "XMLHttpRequest"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            papers = data.get("papers", [])[:max_results]
            
            results = []
            for paper in papers:
                results.append({
                    'id': paper.get('id', ''),
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'authors': paper.get('authors', ''),
                    'url': f"https://arxiv.org/abs/{paper.get('id', '')}",
                    'arxiv_url': f"https://arxiv.org/abs/{paper.get('id', '')}",
                    'pdf_url': f"https://arxiv.org/pdf/{paper.get('id', '')}.pdf",
                    'year': paper.get('year', ''),
                    'similarity_score': paper.get('similarity', 0.0),
                    'source': 'searchthearxiv',
                    'resource_uri': f"arxiv://{paper.get('id', '')}",
                })
            
            return results
        else:
            raise Exception(f"SearchTheArxiv API returned status {response.status_code}")
            
    except Exception as e:
        raise Exception(f"SearchTheArxiv API search failed: {str(e)}")


async def extract_content_jina(url: str, jina_api_key: str) -> Dict[str, Any]:
    """Extract content using Jina AI."""
    if not jina_api_key:
        return {"success": False, "error": "No Jina API key provided"}
        
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {"Authorization": f"Bearer {jina_api_key}"}
        
        response = requests.get(jina_url, headers=headers, timeout=180)
        
        if response.status_code == 200:
            return {
                "success": True,
                "content": response.text,
                "url": url,
                "length": len(response.text),
                "source": "jina_ai"
            }
        else:
            return {
                "success": False,
                "error": f"Jina AI returned status {response.status_code}",
                "url": url
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url
        }


def merge_and_deduplicate_papers(arxiv_papers: List[Dict], searchthearxiv_papers: List[Dict]) -> List[Dict]:
    """Merge and deduplicate papers from both sources, prioritizing ArXiv API data."""
    merged = {}
    
    # Add SearchTheArxiv papers first (lower priority)
    for paper in searchthearxiv_papers:
        paper_id = paper.get('id', '')
        if paper_id:
            merged[paper_id] = paper
    
    # Add ArXiv API papers (higher priority - will overwrite)
    for paper in arxiv_papers:
        paper_id = paper.get('id', '')
        if paper_id:
            # If paper exists from SearchTheArxiv, merge similarity score
            if paper_id in merged:
                existing = merged[paper_id]
                paper['similarity_score'] = existing.get('similarity_score', 0.0)
                paper['searchthearxiv_match'] = True
            merged[paper_id] = paper
    
    # Convert back to list and sort by similarity score  then by date
    result = list(merged.values())
    result.sort(key=lambda x: (x.get('similarity_score', 0.0), x.get('published', '')), reverse=True)
    
    return result


async def handle_hybrid_search(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle hybrid search requests."""
    try:
        query = arguments["query"]
        max_results = min(int(arguments.get("max_results", 10)), settings.MAX_RESULTS)
        search_method = arguments.get("search_method", "hybrid")
        include_content = arguments.get("include_content", False)
        jina_api_key = arguments.get("jina_api_key", "")
        
        date_from = arguments.get("date_from")
        date_to = arguments.get("date_to")
        categories = arguments.get("categories")
        
        results = {"papers": [], "content_extractions": [], "search_methods": []}
        
        # Execute searches based on method
        if search_method in ["arxiv_only", "hybrid"]:
            try:
                arxiv_papers = await search_arxiv_api(
                    query=query,
                    max_results=max_results,
                    date_from=date_from,
                    date_to=date_to,
                    categories=categories
                )
                results["search_methods"].append("arxiv_api")
                if search_method == "arxiv_only":
                    results["papers"] = arxiv_papers
            except Exception as e:
                arxiv_papers = []
                results["arxiv_error"] = str(e)
        else:
            arxiv_papers = []
            
        if search_method in ["searchthearxiv_only", "hybrid"]:
            try:
                searchthearxiv_papers = await search_searchthearxiv_api(
                    query=query,
                    max_results=max_results
                )
                results["search_methods"].append("searchthearxiv_api")
                if search_method == "searchthearxiv_only":
                    results["papers"] = searchthearxiv_papers
            except Exception as e:
                searchthearxiv_papers = []
                results["searchthearxiv_error"] = str(e)
        else:
            searchthearxiv_papers = []
        
        # Merge results for hybrid search
        if search_method == "hybrid":
            results["papers"] = merge_and_deduplicate_papers(arxiv_papers, searchthearxiv_papers)
        
        # Limit final results
        results["papers"] = results["papers"][:max_results]
        
        # Extract content if requested
        if include_content and jina_api_key and results["papers"]:
            top_papers = results["papers"][:min(3, len(results["papers"]))]  # Limit to top 3 for content extraction
            
            extraction_tasks = []
            for paper in top_papers:
                # Try both ArXiv page and PDF
                arxiv_url = paper.get('arxiv_url', '')
                pdf_url = paper.get('pdf_url', '')
                
                if arxiv_url:
                    extraction_tasks.append(extract_content_jina(arxiv_url, jina_api_key))
                if pdf_url:
                    extraction_tasks.append(extract_content_jina(pdf_url, jina_api_key))
            
            if extraction_tasks:
                extractions = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                successful_extractions = [
                    ext for ext in extractions 
                    if isinstance(ext, dict) and ext.get('success')
                ]
                results["content_extractions"] = successful_extractions
        
        # Add metadata
        results.update({
            "total_papers": len(results["papers"]),
            "query": query,
            "search_method": search_method,
            "content_extracted": len(results.get("content_extractions", [])),
            "timestamp": datetime.now().isoformat()
        })
        
        return [
            types.TextContent(type="text", text=json.dumps(results, indent=2))
        ]

    except Exception as e:
        error_result = {
            "error": str(e),
            "query": arguments.get("query", ""),
            "search_method": arguments.get("search_method", "hybrid"),
            "timestamp": datetime.now().isoformat()
        }
        return [
            types.TextContent(type="text", text=json.dumps(error_result, indent=2))
        ]