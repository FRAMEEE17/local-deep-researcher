#!/usr/bin/env python3
"""
Test MCP protocol correctly with proper JSON-RPC sequence.
"""

import json
import subprocess
import time
import signal
import os


def test_mcp_protocol():
    """Test complete MCP protocol sequence."""
    
    print("🧪 Testing Complete MCP Protocol")
    print("=" * 40)
    
    # Change to correct directory
    os.chdir("/home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server")
    
    # Start server process correctly
    process = subprocess.Popen(
        ["uv", "run", "arxiv-mcp-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid  
    )
    
    try:
        time.sleep(2)  # Give server time to start
        
        # Step 1: Initialize (REQUIRED)
        print("📡 Step 1: Initialize...")
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        process.stdin.write(json.dumps(init_msg) + "\n")
        process.stdin.flush()
        
        init_response = process.stdout.readline()
        print(f"✅ Init response: {init_response.strip()}")
        
        # Step 2: Send initialized notification (REQUIRED by MCP spec)
        print("📡 Step 2: Send initialized notification...")
        initialized_msg = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        process.stdin.write(json.dumps(initialized_msg) + "\n")
        process.stdin.flush()
        
        # Step 3: List tools (NOW should work)
        print("📡 Step 3: List tools...")
        tools_msg = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        process.stdin.write(json.dumps(tools_msg) + "\n")
        process.stdin.flush()
        
        tools_response = process.stdout.readline()
        print(f"✅ Tools response: {tools_response.strip()}")
        
        # Parse and show tools
        if tools_response.strip():
            try:
                tools_data = json.loads(tools_response.strip())
                if 'result' in tools_data:
                    tools = tools_data['result'].get('tools', [])
                    print(f"\n🔧 Found {len(tools)} tools:")
                    for tool in tools:
                        print(f"  - {tool.get('name')}: {tool.get('description')}")
                        
                    # Check for hybrid_search
                    hybrid_tool = next((t for t in tools if t.get('name') == 'hybrid_search'), None)
                    if hybrid_tool:
                        print("\n🎉 SUCCESS: hybrid_search tool found!")
                    else:
                        print("\n❌ ERROR: hybrid_search tool missing")
                        
                elif 'error' in tools_data:
                    print(f"❌ Tools list error: {tools_data['error']}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
        
        # Step 4: Test actual tool call
        print("\n📡 Step 4: Test hybrid search call...")
        call_msg = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "hybrid_search",
                "arguments": {
                    "query": "machine learning",
                    "max_results": 2,
                    "search_method": "arxiv_only"
                }
            }
        }
        
        process.stdin.write(json.dumps(call_msg) + "\n")
        process.stdin.flush()
        
        call_response = process.stdout.readline()
        print(f"✅ Call response preview: {call_response[:100]}...")
        
        if call_response.strip():
            try:
                call_data = json.loads(call_response.strip())
                if 'result' in call_data:
                    print("🎉 SUCCESS: Hybrid search call worked!")
                elif 'error' in call_data:
                    print(f"❌ Call error: {call_data['error']}")
            except json.JSONDecodeError:
                print("❌ Call response parse error")
        
        print("\n✅ MCP Protocol Test Complete!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        # Check stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"Server stderr: {stderr_output}")
    
    finally:
        # Clean shutdown
        print("\n🔄 Shutting down server...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            process.terminate()
        process.wait()
        print("✅ Server stopped")


if __name__ == "__main__":
    test_mcp_protocol()