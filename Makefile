# Makefile for ArXiv MCP Server Testing
# =====================================================

.PHONY: help test-direct test-protocol test-mcp test-all clean setup

# Default target
help:
	@echo "🧪 ArXiv MCP Server - Test Suite"
	@echo "==========================================="
	@echo ""
	@echo "Available test targets:"
	@echo ""
	@echo "  test-direct    - Test tools directly (no MCP protocol)"
	@echo "  test-protocol  - Test complete MCP protocol sequence"
	@echo "  test-mcp       - Test MCP server with client integration"
	@echo "  test-all       - Run all tests sequentially"
	@echo ""
	@echo "Setup targets:"
	@echo ""
	@echo "  setup          - Install dependencies and setup environment"
	@echo "  clean          - Clean temporary files and processes"
	@echo ""
	@echo "Usage examples:"
	@echo "  make test-direct    # Test individual tools"
	@echo "  make test-all       # Run comprehensive test suite"

# Test 1: Direct Tools Test (No MCP Protocol)
test-direct:
	@echo "🔧 Running Direct Tools Test..."
	@echo "================================"
	@echo "Testing tools directly without MCP protocol"
	@echo ""
	@cd /home/siamai/deepsad/local-deep-researcher && python test_direct_tools.py
	@echo ""
	@echo "✅ Direct tools test completed!"

# Test 2: MCP Protocol Test (Manual Protocol Implementation)
test-protocol:
	@echo "📡 Running MCP Protocol Test..."
	@echo "==============================="
	@echo "Testing complete MCP JSON-RPC protocol sequence"
	@echo ""
	@cd /home/siamai/deepsad/local-deep-researcher && python test_mcp_protocol.py
	@echo ""
	@echo "✅ MCP protocol test completed!"

# Test 3: MCP Server Test (Client Integration)
test-mcp:
	@echo "🚀 Running MCP Server Test..."
	@echo "============================="
	@echo "Testing MCP server with research pipeline client"
	@echo ""
	@cd /home/siamai/deepsad/local-deep-researcher && python test_mcp_server.py
	@echo ""
	@echo "✅ MCP server test completed!"

# Run all tests sequentially
test-all:
	@echo "🧪 Running Complete Test Suite..."
	@echo "=================================="
	@echo ""
	@make test-direct
	@echo ""
	@echo "⏭️  Moving to next test..."
	@echo ""
	@make test-protocol
	@echo ""
	@echo "⏭️  Moving to final test..."
	@echo ""
	@make test-mcp
	@echo ""
	@echo "🎉 All tests completed successfully!"
	@echo ""
	@echo "📊 Test Summary:"
	@echo "  ✅ Direct Tools Test     - Tools work without MCP"
	@echo "  ✅ MCP Protocol Test     - JSON-RPC protocol works"
	@echo "  ✅ MCP Server Test       - Client integration works"
	@echo ""
	@echo "🚀 Your ArXiv MCP server is ready for production!"

# Setup development environment
setup:
	@echo "⚙️  Setting up development environment..."
	@echo ""
	@echo "📦 Installing ArXiv MCP server dependencies..."
	@cd /home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server && uv add requests aiohttp
	@echo ""
	@echo "🐍 Checking Python environment..."
	@python --version
	@echo ""
	@echo "📋 Verifying key dependencies..."
	@python -c "import sys; print(f'Python path: {sys.executable}')"
	@python -c "import asyncio; print('✅ asyncio available')"
	@python -c "import json; print('✅ json available')"
	@echo ""
	@echo "✅ Setup completed!"

# Clean temporary files and kill any running processes
clean:
	@echo "🧹 Cleaning up..."
	@echo ""
	@echo "🔍 Checking for running MCP server processes..."
	@-pkill -f "arxiv-mcp-server" 2>/dev/null || echo "No MCP server processes found"
	@echo ""
	@echo "🗑️  Removing temporary test files..."
	@-rm -f /tmp/mcp_test_* 2>/dev/null || true
	@echo ""
	@echo "✅ Cleanup completed!"

# Advanced targets
test-debug:
	@echo "🐛 Running Debug Test..."
	@echo "========================"
	@cd /home/siamai/deepsad/local-deep-researcher && python debug_mcp.py

test-simple:
	@echo "📡 Running Simple MCP Test..."
	@echo "============================="
	@cd /home/siamai/deepsad/local-deep-researcher && python test_simple_mcp.py

# Quick health check
health:
	@echo "🩺 Health Check..."
	@echo "=================="
	@echo "Checking ArXiv MCP server components..."
	@cd /home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server && python -c "from src.arxiv_mcp_server.tools import hybrid_search_tool; print(f'✅ Hybrid search tool: {hybrid_search_tool.name}')"
	@cd /home/siamai/deepsad/local-deep-researcher && python -c "from src.research_pipeline.graph import graph; print('✅ Research pipeline compiled')"
	@echo "✅ All components healthy!"

# Show system info
info:
	@echo "ℹ️  System Information"
	@echo "====================="
	@echo "Project: ArXiv MCP Server"
	@echo "Location: /home/siamai/deepsad/local-deep-researcher"
	@echo ""
	@echo "Components:"
	@echo "  📡 ArXiv MCP Server     - /server/arxiv-mcp-server/"
	@echo "  🧠 Research Pipeline   - /src/research_pipeline/"
	@echo "  🧪 Test Suite          - /test_*.py"
	@echo ""
	@echo "Features:"
	@echo "  ✅ ArXiv API Integration"
	@echo "  ✅ SearchTheArxiv Metadata Search"
	@echo "  ✅ Jina AI Content Extraction"
	@echo "  ✅ Hybrid Search Capabilities"
	@echo "  ✅ MCP Protocol Support"
	@echo "  ✅ XLM-RoBERTa Intent Classification"
	@echo ""
	@echo "Usage: make test-all"