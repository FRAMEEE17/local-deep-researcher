import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from subprocess import Popen, PIPE
import os
import tempfile

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers."""
    
    def __init__(self, server_command: str, server_args: List[str], server_env: Optional[Dict[str, str]] = None):
        """
        Initialize MCP client with server configuration.
        
        Args:
            server_command: Command to start the MCP server
            server_args: Arguments for the server command
            server_env: Environment variables for the server
        """
        self.server_command = server_command
        self.server_args = server_args
        self.server_env = server_env or {}
        self.process = None
        
    async def start_server(self):
        """Start the MCP server process."""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.server_env)
            
            # Start the server process
            self.process = Popen(
                [self.server_command] + self.server_args,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                env=env,
                text=True
            )
            
            # Send initialization message
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "langgraph-research-pipeline",
                        "version": "1.0.0"
                    }
                }
            }
            
            await self._send_message(init_message)
            response = await self._receive_message()
            
            if response.get("error"):
                raise Exception(f"MCP server initialization failed: {response['error']}")
            
            # TODO: Send initialized notification (required by MCP spec)
            initialized_message = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self._send_message(initialized_message)
            
            logger.info("MCP server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            return False
    
    async def stop_server(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if not self.process:
            raise Exception("MCP server not started")
            
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        await self._send_message(message)
        response = await self._receive_message()
        
        if response.get("error"):
            raise Exception(f"Tool call failed: {response['error']}")
            
        return response.get("result", {})
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        if not self.process:
            raise Exception("MCP server not started")
            
        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {}
        }
        
        await self._send_message(message)
        response = await self._receive_message()
        
        if response.get("error"):
            raise Exception(f"Failed to list tools: {response['error']}")
            
        return response.get("result", {}).get("tools", [])
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send a message to the MCP server."""
        if not self.process or not self.process.stdin:
            raise Exception("MCP server process not available")
            
        json_message = json.dumps(message) + "\n"
        self.process.stdin.write(json_message)
        self.process.stdin.flush()
    
    async def _receive_message(self) -> Dict[str, Any]:
        """Receive a message from the MCP server."""
        if not self.process or not self.process.stdout:
            raise Exception("MCP server process not available")
            
        line = self.process.stdout.readline()
        if not line:
            raise Exception("No response from MCP server")
            
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from MCP server: {e}")


class ArxivMCPClient(MCPClient):
    """Specialized MCP client for arXiv server."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize arXiv MCP client.
        
        Args:
            storage_path: Path to store downloaded papers
        """
        if not storage_path:
            storage_path = os.path.expanduser("~/.arxiv-mcp-server/papers")
            
        super().__init__(
            server_command="uv",
            server_args=[
                "--directory",
                "/home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server",
                "run",
                "arxiv-mcp-server",
                "--storage-path", storage_path
            ],
            server_env={
                "ARXIV_STORAGE_PATH": storage_path
            }
        )
        
    async def search_papers(self, query: str, max_results: int = 10, 
                          date_from: Optional[str] = None, 
                          categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search arXiv papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            date_from: Start date for search (YYYY-MM-DD)
            categories: arXiv categories to filter by
            
        Returns:
            List of paper metadata
        """
        arguments = {
            "query": query,
            "max_results": max_results
        }
        
        if date_from:
            arguments["date_from"] = date_from
            
        if categories:
            arguments["categories"] = categories
            
        result = await self.call_tool("search_papers", arguments)
        
        # Parse the MCP TextContent response
        if result and "content" in result:
            content = result["content"]
            if content and len(content) > 0:
                # Parse the JSON text from TextContent
                import json
                try:
                    data = json.loads(content[0]["text"])
                    return data.get("papers", [])
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        
        return []
    
    async def download_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Download a paper by ID.
        
        Args:
            paper_id: arXiv paper ID
            
        Returns:
            Download result
        """
        return await self.call_tool("download_paper", {"paper_id": paper_id})
    
    async def read_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Read a downloaded paper's content.
        
        Args:
            paper_id: arXiv paper ID
            
        Returns:
            Paper content
        """
        return await self.call_tool("read_paper", {"paper_id": paper_id})
    
    async def list_papers(self) -> List[Dict[str, Any]]:
        """
        List all downloaded papers.
        
        Returns:
            List of downloaded papers
        """
        result = await self.call_tool("list_papers", {})
        return result.get("papers", [])


# Global client instance
_arxiv_client: Optional[ArxivMCPClient] = None


async def get_arxiv_client() -> ArxivMCPClient:
    """Get or create the global arXiv MCP client."""
    global _arxiv_client
    
    if _arxiv_client is None:
        _arxiv_client = ArxivMCPClient()
        await _arxiv_client.start_server()
        
    return _arxiv_client


async def cleanup_arxiv_client():
    """Cleanup the global arXiv MCP client."""
    global _arxiv_client
    
    if _arxiv_client:
        await _arxiv_client.stop_server()
        _arxiv_client = None