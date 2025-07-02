"""Tool definitions for the arXiv MCP server."""

from .search import search_tool, handle_search
from .download import download_tool, handle_download
from .list_papers import list_tool, handle_list_papers
from .read_paper import read_tool, handle_read_paper
from .hybrid_search import hybrid_search_tool, handle_hybrid_search


__all__ = [
    "search_tool",
    "download_tool",
    "read_tool",
    "list_tool",
    "hybrid_search_tool",
    "handle_search",
    "handle_download",
    "handle_read_paper",
    "handle_list_papers",
    "handle_hybrid_search",
]
