#!/usr/bin/env python3
"""
Debug the MCP server response.
"""

import json
import subprocess
import time


def debug_mcp_server():
    """Debug MCP server responses."""
    
    print("🐛 Debugging MCP Server")
    print("=" * 30)
    
    # Start the server process
    process = subprocess.Popen(
        ["uv", "run", "--directory", "/home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server", "arxiv-mcp-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        time.sleep(1)  # Give server time to start
        
        # Initialize
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "debug-client", "version": "1.0.0"}
            }
        }
        
        process.stdin.write(json.dumps(init_message) + "\n")
        process.stdin.flush()
        
        # Read init response
        init_response = process.stdout.readline()
        print(f"Init response: {init_response.strip()}")
        
        # List tools
        tools_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        process.stdin.write(json.dumps(tools_message) + "\n")
        process.stdin.flush()
        
        # Read tools response
        tools_response = process.stdout.readline()
        print(f"Tools response: {tools_response.strip()}")
        
        # Parse and show tools
        if tools_response.strip():
            try:
                response_data = json.loads(tools_response.strip())
                if 'result' in response_data:
                    tools = response_data['result'].get('tools', [])
                    print(f"\nFound {len(tools)} tools:")
                    for tool in tools:
                        print(f"  - {tool.get('name', 'unnamed')}: {tool.get('description', 'no description')}")
                else:
                    print(f"No result field. Full response: {response_data}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Check stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"Stderr: {stderr_output}")
    
    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    debug_mcp_server()