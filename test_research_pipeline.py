#!/usr/bin/env python3
"""
Test Script for LangGraph Research Pipeline
Tests the complete research workflow with ArXiv MCP integration
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from research_pipeline.graph import app
from research_pipeline.configuration import Configuration
from research_pipeline.state import ResearchStateInput
from langchain_core.runnables import RunnableConfig

class ResearchPipelineTest:
    """Test suite for the research pipeline."""
    
    def __init__(self):
        self.config = Configuration()
        self.test_results = []
    
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
    
    async def test_basic_configuration(self):
        """Test basic configuration loading."""
        try:
            config = Configuration()
            assert hasattr(config, 'max_web_research_loops')
            assert hasattr(config, 'local_llm')
            assert hasattr(config, 'llm_provider')
            self.log_test("Basic Configuration", "PASS", f"Model: {config.local_llm}")
        except Exception as e:
            self.log_test("Basic Configuration", "FAIL", str(e))
    
    async def test_intent_classification(self):
        """Test intent classification functionality."""
        try:
            from research_pipeline.intent_classifier import classify_query_intent
            
            # Test academic query
            academic_query = "transformer neural networks attention mechanism"
            intent, confidence = classify_query_intent(academic_query)
            self.log_test("Intent Classification - Academic", "PASS", 
                         f"Intent: {intent}, Confidence: {confidence:.3f}")
            
            # Test web query
            web_query = "latest news about artificial intelligence"
            intent, confidence = classify_query_intent(web_query)
            self.log_test("Intent Classification - Web", "PASS", 
                         f"Intent: {intent}, Confidence: {confidence:.3f}")
            
        except ImportError:
            self.log_test("Intent Classification", "SKIP", "Intent classifier not available")
        except Exception as e:
            self.log_test("Intent Classification", "FAIL", str(e))
    
    async def test_arxiv_mcp_connection(self):
        """Test ArXiv MCP server connection."""
        try:
            from research_pipeline.arxiv_http_client import ArxivMCPOClient
            
            client = ArxivMCPOClient()
            
            # Test basic search
            result = await client.search_papers("transformer", max_results=2)
            
            if result and len(result) > 0:
                self.log_test("ArXiv MCP Connection", "PASS", 
                             f"Found {len(result)} papers")
            else:
                self.log_test("ArXiv MCP Connection", "FAIL", "No results returned")
                
        except Exception as e:
            self.log_test("ArXiv MCP Connection", "FAIL", str(e))
    
    async def test_search_engines(self):
        """Test search engines functionality."""
        try:
            from research_pipeline.search_engines import create_search_engines
            
            search_engines = create_search_engines()
            
            # Test ArXiv search strategy
            test_query = "machine learning optimization"
            result = await search_engines.execute_search_strategy(
                strategy="arxiv_search",
                query=test_query,
                max_results=2
            )
            
            if result:
                self.log_test("Search Engines - ArXiv", "PASS", 
                             f"Strategy executed successfully")
            else:
                self.log_test("Search Engines - ArXiv", "FAIL", "No results")
                
        except ImportError:
            self.log_test("Search Engines", "SKIP", "Search engines not available")
        except Exception as e:
            self.log_test("Search Engines", "FAIL", str(e))
    
    async def test_langgraph_workflow(self):
        """Test the complete LangGraph workflow."""
        try:
            # Create test input
            input_data = ResearchStateInput(
                research_topic="transformer neural networks for natural language processing"
            )
            
            # Configure the run
            config = RunnableConfig(
                configurable={
                    "max_web_research_loops": 1,  # Limit for testing
                    "local_llm": self.config.local_llm,
                    "llm_provider": self.config.llm_provider,
                }
            )
            
            # Run the workflow
            print("\n🔄 Running LangGraph workflow...")
            start_time = time.time()
            
            result = await app.ainvoke(input_data, config=config)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Check results
            if result and hasattr(result, 'final_summary'):
                self.log_test("LangGraph Workflow", "PASS", 
                             f"Completed in {duration:.2f}s")
                
                # Print summary of results
                print(f"\n📊 Workflow Results:")
                print(f"   Research Topic: {result.research_topic}")
                print(f"   Search Intent: {getattr(result, 'search_intent', 'Unknown')}")
                print(f"   ArXiv Results: {len(getattr(result, 'arxiv_results', []))}")
                print(f"   Web Results: {len(getattr(result, 'web_results', []))}")
                print(f"   Final Summary Length: {len(result.final_summary) if result.final_summary else 0}")
                
            else:
                self.log_test("LangGraph Workflow", "FAIL", "No final summary generated")
                
        except Exception as e:
            self.log_test("LangGraph Workflow", "FAIL", str(e))
    
    async def test_specific_paper_query(self):
        """Test handling of specific ArXiv paper queries."""
        try:
            # Test with the paper the user asked about
            input_data = ResearchStateInput(
                research_topic="explain arxiv paper 2410.21338v2 about large language models"
            )
            
            config = RunnableConfig(
                configurable={
                    "max_web_research_loops": 1,
                    "local_llm": self.config.local_llm,
                    "llm_provider": self.config.llm_provider,
                }
            )
            
            print("\n🔄 Testing specific paper query...")
            start_time = time.time()
            
            result = await app.ainvoke(input_data, config=config)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result:
                self.log_test("Specific Paper Query", "PASS", 
                             f"Processed in {duration:.2f}s")
                
                # Check if it found the specific paper
                arxiv_results = getattr(result, 'arxiv_results', [])
                found_target_paper = any('2410.21338' in str(paper) for paper in arxiv_results)
                
                if found_target_paper:
                    print("   ✅ Found target paper 2410.21338v2")
                else:
                    print("   ⚠️ Target paper not found in results")
                    
            else:
                self.log_test("Specific Paper Query", "FAIL", "No result returned")
                
        except Exception as e:
            self.log_test("Specific Paper Query", "FAIL", str(e))
    
    async def test_mcpo_direct_connection(self):
        """Test direct MCPO connection."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Test MCPO health
                async with session.get("http://localhost:9937/docs") as response:
                    if response.status == 200:
                        self.log_test("MCPO Health Check", "PASS", "MCPO server is running")
                    else:
                        self.log_test("MCPO Health Check", "FAIL", f"Status: {response.status}")
                
                # Test ArXiv endpoint
                search_data = {
                    "query": "transformer neural networks",
                    "max_results": 2
                }
                
                async with session.post(
                    "http://localhost:9937/arxiv-mcp-server/search_papers",
                    json=search_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers_count = len(data.get('papers', []))
                        self.log_test("MCPO ArXiv Search", "PASS", 
                                     f"Found {papers_count} papers")
                    else:
                        self.log_test("MCPO ArXiv Search", "FAIL", 
                                     f"Status: {response.status}")
                        
        except Exception as e:
            self.log_test("MCPO Direct Connection", "FAIL", str(e))
    
    def print_summary(self):
        """Print test summary."""
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        skipped = sum(1 for r in self.test_results if r['status'] == 'SKIP')
        
        print(f"\n📊 Test Summary")
        print(f"===============")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Skipped: {skipped}")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        # Save results to file
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")

async def main():
    """Run all tests."""
    print("🧪 Research Pipeline Test Suite")
    print("================================")
    
    tester = ResearchPipelineTest()
    
    # Run all tests
    await tester.test_basic_configuration()
    await tester.test_intent_classification()
    await tester.test_mcpo_direct_connection()
    await tester.test_arxiv_mcp_connection()
    await tester.test_search_engines()
    await tester.test_langgraph_workflow()
    await tester.test_specific_paper_query()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())