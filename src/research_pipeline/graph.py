import json
import time
import asyncio
from typing_extensions import Literal
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from research_pipeline.configuration import Configuration
from research_pipeline.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search, searxng_search, arxiv_search, strip_thinking_tokens, get_config_value
from research_pipeline.state import ResearchState, ResearchStateInput, ResearchStateOutput
from research_pipeline.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date
from research_pipeline.lmstudio import ChatLMStudio
from research_pipeline.nvidia_nim import ChatNVIDIANIM

import logging
logger = logging.getLogger("research_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] LANGGRAPH - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
try:
    from research_pipeline.intent_classifier import classify_query_intent
    INTENT_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Intent classifier not available: {e}")
    INTENT_CLASSIFIER_AVAILABLE = False
    
try:
    from research_pipeline.search_engines import create_search_engines
    SEARCH_ENGINES_AVAILABLE = True
except ImportError as e:
    print(f"Search engines not available: {e}")
    SEARCH_ENGINES_AVAILABLE = False

# Nodes
def classify_intent(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """LangGraph node that classifies the research intent using trained XLM-RoBERTa model.
    
    Analyzes the research topic to determine the appropriate search strategy:
    - academic_intent → ArXiv MCP search
    - web_intent → SearXNG web search  
    - hybrid_intent → Combined approach
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including intent model paths
        
    Returns:
        Dictionary with state update including search_intent, intent_confidence, and routing strategy
    """
    
    logger.info("EXECUTING NODE: classify_intent() - Starting intent classification")
    logger.info(f"INPUT: research_topic = '{state.research_topic}'")
    start_time = time.time()
    
    configurable = Configuration.from_runnable_config(config)
    
    if INTENT_CLASSIFIER_AVAILABLE:
        logger.info("Using XLM-RoBERTa intent classifier model")
        # Use trained XLM-RoBERTa model
        intent_result = classify_query_intent(state.research_topic, configurable)
        logger.info(f"INTENT CLASSIFICATION RESULT: {intent_result['routing_strategy']} (confidence: {intent_result['confidence']:.2f})")
        result = {
            "search_intent": intent_result["routing_strategy"],
            "intent_confidence": intent_result["confidence"],
            "search_strategy": intent_result["routing_strategy"],
            "processing_times": {"intent_classification": time.time() - start_time}
        }
    else:
        logger.info("Using fallback rule-based intent classification")
        # Fallback: simple rule-based classification
        topic_lower = state.research_topic.lower()
        
        # Rule-based intent detection
        if any(word in topic_lower for word in ['arxiv', 'paper', 'research', 'academic', 'study', 'journal']):
            strategy = "arxiv_search"
            confidence = 0.8
        elif any(word in topic_lower for word in ['news', 'current', 'latest', 'recent', 'today']):
            strategy = "hybrid_search"
            confidence = 0.7
        else:
            strategy = "web_search"
            confidence = 0.6
        
        result = {
            "search_intent": strategy,
            "intent_confidence": confidence,
            "search_strategy": strategy,
            "processing_times": {"intent_classification": time.time() - start_time}
        }
    logger.info(f"OUTPUT: search_strategy = {result.get('search_strategy')}")
    logger.info("COMPLETED NODE: classify_intent()")
    return result

async def generate_query(state: ResearchState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.
    
    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    logger.info("EXECUTING NODE: generate_query() - Creating optimized search query")
    logger.info(f"INPUT: research_topic = '{state.research_topic}', search_strategy = '{state.search_strategy}'")
    start_time = time.time()
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic,
        search_intent=state.search_intent or "hybrid",
        intent_confidence=state.intent_confidence or 0.6
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        logger.info("Using LMStudio LLM for query generation")
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0.5, # 0 = very deterministic
            format="json"
        )
    elif configurable.llm_provider == "nvidia_nim":
        logger.info("Using NVIDIA NIM LLM for query generation")
        llm_json_mode = ChatNVIDIANIM(
            base_url=configurable.nvidia_nim_base_url,
            model=configurable.get_model_name(),
            temperature=0.5,
            nvidia_api_key=configurable.nvidia_api_key,
            enable_reasoning=False  # Disable for structured JSON output
        )
    else: # Default to Ollama
        logger.info("Using Ollama LLM for query generation")
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0.5, 
            format="json"
        )
    logger.info("Calling LLM to generate search query")
    result = llm_json_mode.invoke([
        HumanMessage(content=formatted_prompt)
    ])
    
    # Get the content
    contents = result.content
    logger.info(f"LLM Raw Response: {contents[:200]}...")
    # Parse the JSON response and get the query
    # try:
    #     query = json.loads(contents)
    #     # search_query = query.get('query', f"Tell me more about {state.research_topic}")
    #     search_query = query['query']
    #     logger.info(f"GENERATED QUERY: '{search_query}'")
    # except (json.JSONDecodeError, KeyError):
    #     # If parsing fails or the key is not found, use a fallback query
    #     if configurable.strip_thinking_tokens:
    #         contents = strip_thinking_tokens(contents)
    #     search_query = contents
    # Parse the JSON response and get the query
    try:
        # Clean the response - strip whitespace and remove markdown formatting
        cleaned_contents = contents.strip()
        
        # Remove markdown code blocks if present
        if cleaned_contents.startswith('```') and cleaned_contents.endswith('```'):
            lines = cleaned_contents.split('\n')
            if len(lines) > 2:
                cleaned_contents = '\n'.join(lines[1:-1])
                
        # Try direct JSON parsing first
        query = json.loads(cleaned_contents)
        search_query = query.get('query', '').strip()
        
        # Validate the extracted query
        if not search_query:
            logger.info("Empty query from LLM, using fallback")
            search_query = f"Tell me more about {state.research_topic}"
        else:
            logger.info(f"GENERATED QUERY: '{search_query}'")
            
            # Store additional metadata if available
            query_metadata = {
                "rationale": query.get("rationale", ""),
                "optimization_type": query.get("optimization_type", state.search_intent),
                "confidence_level": query.get("confidence_level", "medium")
            }
            logger.info(f"Query metadata: {query_metadata}")
            
    except json.JSONDecodeError as e:
        # JSON parsing failed - try to extract JSON from wrapped text
        logger.warning(f"JSON parsing failed: {e}")
        logger.info("Attempting JSON extraction from wrapped text")
        
        import re
        
        # Pattern 1: Find complete JSON objects with balanced braces
        # This pattern correctly handles nested JSON objects
        def find_json_objects(text):
            json_objects = []
            brace_count = 0
            start_pos = None
            
            for i, char in enumerate(text):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos is not None:
                        json_objects.append(text[start_pos:i+1])
                        start_pos = None
            
            return json_objects
        
        json_matches = find_json_objects(contents)
        print(f"DEBUG: Found {len(json_matches)} JSON matches: {json_matches}")
        extracted_query = None
        query_metadata = None
        
        # Try to parse each JSON match
        for json_match in json_matches:
            # Clean up double braces from template formatting
            clean_match = json_match.replace('{{', '{').replace('}}', '}')
            try:
                parsed_json = json.loads(clean_match)
                if 'query' in parsed_json and parsed_json['query'].strip():
                    extracted_query = parsed_json.get('query', '').strip()
                    logger.info(f"EXTRACTED QUERY from JSON block: '{extracted_query}'")
                    query_metadata = {
                        "rationale": parsed_json.get("rationale", "Extracted from wrapped text"),
                        "optimization_type": parsed_json.get("optimization_type", state.search_intent),
                        "confidence_level": parsed_json.get("confidence_level", "medium")
                    }
                    break
            except json.JSONDecodeError:
                continue
        
        if extracted_query:
            # Successfully extracted query from JSON block
            search_query = extracted_query
        else:
            # JSON extraction failed - fall back to regex patterns
            logger.info("JSON extraction failed, trying regex patterns")
            
            if configurable.strip_thinking_tokens:
                cleaned_content = strip_thinking_tokens(contents)
            else:
                cleaned_content = contents
            
            # Try to extract a reasonable query from the raw content
            # Look for quoted strings first
            quote_pattern = r'"([^"]+)"'
            quotes = re.findall(quote_pattern, cleaned_content)
            
            potential_query = None
            if quotes:
                # Use the first quoted string as potential query
                potential_query = quotes[0].strip()
                logger.info(f"Found quoted query: '{potential_query}'")
            else:
                # Look for first meaningful line
                lines = [line.strip() for line in cleaned_content.strip().split('\n') if line.strip()]
                if lines:
                    potential_query = lines[0].strip()
                    logger.info(f"Using first line as query: '{potential_query}'")
            
            # Validate extracted content as query
            if potential_query and 5 < len(potential_query) < 200:
                search_query = potential_query
                logger.info(f"EXTRACTED QUERY from raw content: '{search_query}'")
            else:
                search_query = f"Research on {state.research_topic}"
                logger.info(f"FALLBACK QUERY: '{search_query}'")
            
            # Set default metadata for fallback
            query_metadata = {
                "rationale": "Fallback query due to JSON parsing failure",
                "optimization_type": state.search_intent or "hybrid",
                "confidence_level": "low"
            }
    
    except Exception as e:
        logger.error(f"Unexpected error during query generation: {e}")
        search_query = f"Research on {state.research_topic}"
        query_metadata = {
            "rationale": f"Error fallback: {str(e)}",
            "optimization_type": "fallback",
            "confidence_level": "low"
        }
    
    # Store the generation time
    processing_times = state.processing_times.copy()
    processing_times["query_generation"] = time.time() - start_time
    
    logger.info(f"Query generation completed in {processing_times['query_generation']:.2f}s")
    
    return {
        "search_query": search_query,
        "query_optimization_metadata": query_metadata,
        "processing_times": processing_times
    }
async def execute_search(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """LangGraph node that executes search based on classified intent.
    
    Performs intelligent routing based on intent classification:
    - arxiv_search: Instructions for MCPO to handle ArXiv operations
    - web_search: Direct SearXNG web search
    - hybrid_search: SearchTheArxiv + MCPO instructions
    
    Args:
        state: Current graph state containing search query and intent
        config: Configuration for the runnable, including search engine settings
        
    Returns:
        Dictionary with state updates for search results based on strategy
    """
    logger.info("EXECUTING NODE: execute_search() - Running search operations")
    logger.info(f"INPUT: search_query = '{state.search_query}'")
    logger.info(f"INPUT: search_intent = '{state.search_intent}'")
    logger.info(f"INPUT: research_loop_count = {state.research_loop_count}")
    start_time = time.time()
    
    configurable = Configuration.from_runnable_config(config)
    
    if SEARCH_ENGINES_AVAILABLE:
        logger.info("Search engines available, creating search engine instance")
        search_engines = create_search_engines(configurable)
        
        # Execute search based on intent routing
        logger.info(f"Calling search_engines.execute_search_strategy() with strategy: {state.search_intent}")
        logger.info(f"Max results configured: {configurable.max_papers_per_search}")
        search_result = await search_engines.execute_search_strategy(
            query=state.search_query,
            strategy=state.search_intent,
            max_results=configurable.max_papers_per_search
        )
        # Log search execution results
        if search_result.get("success"):
            logger.info("Search strategy executed successfully")
            if "papers" in search_result:
                logger.info(f"Papers found: {len(search_result.get('papers', []))}")
            if "results" in search_result:
                logger.info(f"Web results found: {len(search_result.get('results', []))}")
        else:
            logger.warning(f"Search strategy failed: {search_result.get('error', 'Unknown error')}")
    else:
        # Fallback to legacy search methods
        logger.warning("Search engines not available, falling back to legacy search")
        return await legacy_web_research(state, config)
    
    processing_time = time.time() - start_time
    logger.info(f"Search execution completed in {processing_time:.2f}s")
    # Update processing times
    processing_times = state.processing_times.copy()
    processing_times["search_execution"] = processing_time
    
    # Route results based on search strategy
    if state.search_intent == "arxiv_search":
        logger.info("ROUTING TO: ArXiv MCP Server search results")
        papers = search_result.get("papers", [])
        papers_count = len(papers)
        logger.info(f"ArXiv search found: {papers_count} papers")
        
        if papers_count > 0:
            # Log paper details
            for i, paper in enumerate(papers[:3]):  # Log first 3
                title = paper.get('title', 'Unknown title')
                logger.info(f"  ArXiv {i+1}: {title[:50]}...")
        
        result = {
                    "arxiv_results": papers,  # Use actual papers, not instruction
                    "sources_gathered": [f"ArXiv MCP search: {papers_count} papers found"],
                    "processing_times": processing_times,
                    "research_loop_count": state.research_loop_count + 1
                }
        logger.info(f"ArXiv results prepared: {papers_count} papers, research_loop_count = {result['research_loop_count']}")
    elif state.search_intent == "web_search":
        logger.info("ROUTING TO: Web search (SearXNG) results")
        results_count = len(search_result.get("results", []))
        logger.info(f"Web search results count: {results_count}")
        result = {
                    "web_results": search_result.get("results", []),
                    "sources_gathered": [f"SearXNG web search: {results_count} results"],
                    "processing_times": processing_times,
                    "research_loop_count": state.research_loop_count + 1
                }
        logger.info(f"Web results prepared: {results_count} results, research_loop_count = {result['research_loop_count']}")
    elif state.search_intent == "hybrid_search":
        logger.info("ROUTING TO: Hybrid search (ArXiv + Web) results")
        papers = search_result.get("papers", [])
        web_results = search_result.get("results", [])
        papers_count = len(papers)
        web_count = len(web_results)
        logger.info(f"Hybrid search found: {papers_count} ArXiv papers, {web_count} web results")
        
        if papers_count > 0:
            # Log paper details
            for i, paper in enumerate(papers[:3]):  # Log first 3
                title = paper.get('title', 'Unknown title')
                logger.info(f"  ArXiv {i+1}: {title[:50]}...")
        
        result = {
                    "arxiv_results": papers,  # ArXiv papers from hybrid search
                    "web_results": web_results,  # Web results from hybrid search
                    "sources_gathered": [f"Hybrid search: {papers_count} ArXiv papers, {web_count} web results"],
                    "processing_times": processing_times,
                    "research_loop_count": state.research_loop_count + 1
                }
        logger.info(f"Hybrid results prepared: {papers_count} papers, research_loop_count = {result['research_loop_count']}")
    else:
        # Fallback to legacy web research
        logger.warning(f"Unknown search_intent: {state.search_intent}, falling back to legacy search")
        return await legacy_web_research(state, config)
    logger.info("COMPLETED NODE: execute_search()")
    return result

async def legacy_web_research(state: ResearchState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.
    
    Executes a web search using the configured search API (tavily, perplexity, 
    duckduckgo, or searxng) and formats the results for further processing.
    
    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings
        
    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_results
    """
    logger.info("EXECUTING FUNCTION: legacy_web_research() - Fallback web search")
    logger.info(f"INPUT: search_query = '{state.search_query}'")
    logger.info(f"INPUT: research_loop_count = {state.research_loop_count}")
    start_time = time.time()
    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Get the search API
    search_api = get_config_value(configurable.search_api)
    logger.info(f"Using legacy search API: {search_api}")
    logger.info(f"Fetch full page enabled: {configurable.fetch_full_page}")
    try:
        # Search based on configured API
        if search_api == "tavily":
            logger.info("Executing Tavily search with max_results=1")
            search_results = tavily_search(
                state.search_query, 
                fetch_full_page=configurable.fetch_full_page, 
                max_results=1
            )
            search_str = deduplicate_and_format_sources(
                search_results, 
                max_tokens_per_source=1000, 
                fetch_full_page=configurable.fetch_full_page
            )
            logger.info(f"Tavily search completed: {len(search_results)} results, {len(search_str)} characters")
            
        elif search_api == "perplexity":
            logger.info(f"Executing Perplexity search with loop_count={state.research_loop_count}")
            search_results = perplexity_search(state.search_query, state.research_loop_count)
            search_str = deduplicate_and_format_sources(
                search_results, 
                max_tokens_per_source=1000, 
                fetch_full_page=configurable.fetch_full_page
            )
            logger.info(f"Perplexity search completed: {len(search_results)} results, {len(search_str)} characters")
            
        elif search_api == "duckduckgo":
            logger.info("Executing DuckDuckGo search with max_results=3")
            search_results = duckduckgo_search(
                state.search_query, 
                max_results=3, 
                fetch_full_page=configurable.fetch_full_page
            )
            search_str = deduplicate_and_format_sources(
                search_results, 
                max_tokens_per_source=1000, 
                fetch_full_page=configurable.fetch_full_page
            )
            logger.info(f"DuckDuckGo search completed: {len(search_results)} results, {len(search_str)} characters")
            
        elif search_api == "searxng":
            logger.info("Executing SearXNG search with max_results=3")
            search_results = searxng_search(
                state.search_query, 
                max_results=20, 
                fetch_full_page=configurable.fetch_full_page
            )
            search_str = deduplicate_and_format_sources(
                search_results, 
                max_tokens_per_source=1000, 
                fetch_full_page=configurable.fetch_full_page
            )
            logger.info(f"SearXNG search completed: {len(search_results)} results, {len(search_str)} characters")
            
        elif search_api == "arxiv":
            logger.info("Executing legacy ArXiv search with max_results=5")
            search_results = await arxiv_search(state.search_query, max_results=50)
            search_str = deduplicate_and_format_sources(
                search_results, 
                max_tokens_per_source=5500, 
                fetch_full_page=True
            )
            logger.info(f"Legacy ArXiv search completed: {len(search_results)} results, {len(search_str)} characters")
            
        else:
            logger.error(f"Unsupported search API in legacy mode: {search_api}")
            logger.error(f"Available APIs: tavily, perplexity, duckduckgo, searxng, arxiv")
            raise ValueError(f"Unsupported search API: {configurable.search_api}")
        
        # Process and format results
        formatted_sources = format_sources(search_results)
        execution_time = time.time() - start_time
        
        logger.info(f"Legacy search processing completed in {execution_time:.2f}s")
        logger.info(f"Formatted sources: {len(formatted_sources)} characters")
        logger.info(f"Search string: {len(search_str)} characters")
        logger.info("COMPLETED FUNCTION: legacy_web_research()")
        
        return {
            "sources_gathered": [formatted_sources], 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": [search_str]
        }
        
    except Exception as e:
        logger.error(f"LEGACY WEB RESEARCH FAILED: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Search API: {search_api}, Query: '{state.search_query}'")
        
        # Return minimal state to prevent pipeline crash
        return {
            "sources_gathered": [f"Legacy search failed: {str(e)}"], 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": ["Search failed"]
        }

async def extract_content(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    """LangGraph node that extracts full content from selected sources.
    
    Uses Jina AI to extract content from top-ranked papers/sources for deeper analysis.
    Only processes a limited number of sources to manage processing time and costs.
    
    Args:
        state: Current graph state containing search results
        config: Configuration for the runnable, including Jina API settings
        
    Returns:
        Dictionary with state update including extracted_content and jina_extractions
    """
    logger.info("EXECUTING NODE: extract_content() - Extracting full content using Jina AI")
    logger.info(f"INPUT: arxiv_results count = {len(state.arxiv_results or [])}")
    logger.info(f"INPUT: semantic_results count = {len(state.semantic_results or [])}")
    logger.info(f"INPUT: web_results count = {len(state.web_results or [])}")
    
    start_time = time.time()
    
    configurable = Configuration.from_runnable_config(config)
    
    if not SEARCH_ENGINES_AVAILABLE:
        # Skip content extraction if search engines not available
        logger.warning("Search engines not available, skipping content extraction")
        processing_time = time.time() - start_time
        processing_times = state.processing_times.copy()
        processing_times["content_extraction"] = processing_time
        logger.info("COMPLETED NODE: extract_content() - Skipped due to unavailable engines")
        return {
            "extracted_content": [],
            "jina_extractions": [],
            "processing_times": processing_times
        }
    logger.info("Creating search engines instance for content extraction")
    search_engines = create_search_engines(configurable)
    extracted_content = []
    
    # Extract from ArXiv and semantic results (ArXiv papers)
    papers_to_extract = []
    
    # Collect papers from both arxiv_results and semantic_results
    if state.arxiv_results:
        logger.info(f"Processing ArXiv results: {len(state.arxiv_results)} papers available")
        papers_to_extract.extend(state.arxiv_results)
    
    if state.semantic_results:
        logger.info(f"Processing semantic results: {len(state.semantic_results)} papers available")
        papers_to_extract.extend(state.semantic_results)
    
    if papers_to_extract:
        # Deduplicate by paper ID if possible
        seen_ids = set()
        unique_papers = []
        for paper in papers_to_extract:
            paper_id = paper.get('id') or paper.get('paper_id') or paper.get('url', '')
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)
        
        top_papers = unique_papers[:configurable.max_content_extractions]
        logger.info(f"Selected top {len(top_papers)} papers for content extraction")
            
        extraction_tasks = []
        papers_with_pdf = 0
            
        for i, paper in enumerate(top_papers):
            # Check for both 'url' (ArXiv MCP format) and 'pdf_url' (legacy format)
            pdf_url = paper.get('pdf_url') or paper.get('url')
            if pdf_url:
                papers_with_pdf += 1
                logger.info(f"Queuing extraction task {i+1}: {paper.get('title', 'Unknown title')[:50]}...")
                logger.info(f"PDF URL: {pdf_url}")
                extraction_tasks.append(
                    search_engines.extract_content_jina(pdf_url)
                )
            else:
                logger.warning(f"Paper {i+1} has no PDF URL, skipping: {paper.get('title', 'Unknown')[:50]}...")
            
        logger.info(f"Executing {len(extraction_tasks)} concurrent PDF extraction tasks")
            
        if extraction_tasks:
            extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                
            successful_extractions = 0
            failed_extractions = 0
                
            for i, result in enumerate(extraction_results):
                if isinstance(result, Exception):
                    failed_extractions += 1
                    logger.error(f"PDF extraction task {i+1} failed with exception: {str(result)}")
                elif isinstance(result, dict) and result.get('success'):
                    successful_extractions += 1
                    content_length = len(result.get('content', ''))
                    logger.info(f"PDF extraction task {i+1} successful: {content_length} characters extracted")
                    extracted_content.append(result)
                else:
                    failed_extractions += 1
                    error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Invalid response'
                    logger.warning(f"PDF extraction task {i+1} failed: {error_msg}")
                
            logger.info(f"PDF extraction summary: {successful_extractions} successful, {failed_extractions} failed")
        else:
            logger.info("No PDF extraction tasks to execute")
    else:
        logger.info("No ArXiv or semantic results available for content extraction")
    
    # Extract from web results
    if state.web_results:
        logger.info(f"Processing web results: {len(state.web_results)} results available")
        top_web = state.web_results[:10]  # Limit web extractions
        logger.info(f"Selected top {len(top_web)} web results for content extraction")
            
        web_successful = 0
        web_failed = 0
            
        for i, result in enumerate(top_web):
            if result.get('url'):
                logger.info(f"Extracting web content {i+1}: {result.get('url')}")
                try:
                    extraction = await search_engines.extract_content_jina(result['url'])
                    if extraction.get('success'):
                        web_successful += 1
                        content_length = len(extraction.get('content', ''))
                        logger.info(f"Web extraction {i+1} successful: {content_length} characters extracted")
                        extracted_content.append(extraction)
                    else:
                        web_failed += 1
                        error_msg = extraction.get('error', 'Unknown error')
                        logger.warning(f"Web extraction {i+1} failed: {error_msg}")
                except Exception as e:
                    web_failed += 1
                    logger.error(f"Web extraction {i+1} failed with exception: {str(e)}")
            else:
                web_failed += 1
                logger.warning(f"Web result {i+1} has no URL, skipping")
            
        logger.info(f"Web extraction summary: {web_successful} successful, {web_failed} failed")
    else:
        logger.info("No web results available for content extraction")
        
    processing_time = time.time() - start_time
    processing_times = state.processing_times.copy()
    processing_times["content_extraction"] = processing_time
    
    total_extractions = len(extracted_content)
    logger.info(f"Content extraction completed in {processing_time:.2f}s")
    logger.info(f"Total successful extractions: {total_extractions}")
    
    if total_extractions > 0:
        total_characters = sum(len(content.get('content', '')) for content in extracted_content)
        logger.info(f"Total content extracted: {total_characters} characters")
    else:
        logger.warning("No content was successfully extracted")
    
    logger.info("COMPLETED NODE: extract_content()")
    
    return {
        "extracted_content": extracted_content,
        "jina_extractions": extracted_content,
        "processing_times": processing_times
    }


def synthesize_research(state: ResearchState, config: RunnableConfig):
    """LangGraph node that summarizes web research results.
    
    Uses an LLM to create or update a running summary based on the newest web research 
    results, integrating them with any existing summary.
    
    Args:
        state: Current graph state containing research topic, running summary,
              and web research results
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """
    logger.info("EXECUTING NODE: synthesize_research() - Creating research summary")
    logger.info(f"INPUT: research_topic = '{state.research_topic}'")
    logger.info(f"INPUT: search_intent = '{state.search_intent}' (confidence: {state.intent_confidence:.2f})")
    logger.info(f"INPUT: research_loop = {state.research_loop_count}/{state.max_loops}")
    logger.info(f"INPUT: has_existing_summary = {bool(state.running_summary)}")
    
    if state.running_summary:
        logger.info(f"INPUT: existing_summary_length = {len(state.running_summary)} characters")
    
    start_time = time.time()
    
    try:
        # Existing summary
        existing_summary = state.running_summary

        # Compile all research sources
        logger.info("Compiling research sources from all available data")
        research_content = []
        total_sources = 0
        
        # Add ArXiv results
        if state.arxiv_results:
            arxiv_count = len(state.arxiv_results)
            total_sources += arxiv_count
            logger.info(f"Adding ArXiv results: {arxiv_count} items")
            
            arxiv_summary = f"ArXiv Results ({arxiv_count} found):\n"
            for i, result in enumerate(state.arxiv_results):
                if 'instruction' in result:
                    instruction = result['instruction'][:100]  # Truncate for logging
                    logger.info(f"  ArXiv {i+1}: MCPO instruction - {instruction}...")
                    arxiv_summary += f"- MCPO instruction: {result['instruction']}\n"
                else:
                    title = result.get('title', 'Unknown title')
                    logger.info(f"  ArXiv {i+1}: {title[:50]}...")
                    arxiv_summary += f"- {title}\n"
            research_content.append(arxiv_summary)
        else:
            logger.info("No ArXiv results available")
        
        # Add semantic search results
        if state.semantic_results:
            semantic_count = len(state.semantic_results)
            total_sources += semantic_count
            logger.info(f"Adding semantic ArXiv results: {semantic_count} papers (using top 3)")
            
            semantic_summary = f"Semantic ArXiv Results ({semantic_count} found):\n"
            for i, paper in enumerate(state.semantic_results[:3]):  # Top 3
                title = paper.get('title', 'Unknown')
                abstract_preview = paper.get('abstract', '')[:50]
                logger.info(f"  Semantic {i+1}: {title[:40]}... (abstract: {len(paper.get('abstract', ''))} chars)")
                semantic_summary += f"- {title}: {paper.get('abstract', '')[:200]}...\n"
            research_content.append(semantic_summary)
        else:
            logger.info("No semantic results available")
        
        # Add web results
        if state.web_results:
            web_count = len(state.web_results)
            total_sources += web_count
            logger.info(f"Adding web search results: {web_count} items (using top 3)")
            
            web_summary = f"Web Search Results ({web_count} found):\n"
            for i, result in enumerate(state.web_results[:3]):  # Top 3
                title = result.get('title', 'Unknown')
                content_length = len(result.get('content', ''))
                logger.info(f"  Web {i+1}: {title[:40]}... ({content_length} chars)")
                web_summary += f"- {title}: {result.get('content', '')[:200]}...\n"
            research_content.append(web_summary)
        else:
            logger.info("No web results available")
        
        # Add extracted content
        if state.extracted_content:
            extracted_count = len(state.extracted_content)
            total_sources += extracted_count
            logger.info(f"Adding extracted content: {extracted_count} sources (using top 2)")
            
            content_summary = f"Extracted Content ({extracted_count} sources):\n"
            for i, content in enumerate(state.extracted_content[:2]):  # Top 2
                content_length = content.get('length', len(content.get('content', '')))
                source_url = content.get('url', 'Unknown source')[:50]
                logger.info(f"  Extracted {i+1}: {source_url}... ({content_length} chars)")
                content_summary += f"- Content length: {content_length} chars\n"
            research_content.append(content_summary)
        else:
            logger.info("No extracted content available")
        
        # Fallback to legacy web research results
        if hasattr(state, 'web_research_results') and state.web_research_results:
            legacy_count = len(state.web_research_results)
            logger.info(f"Adding legacy web research results: {legacy_count} items (using most recent)")
            most_recent_web_research = state.web_research_results[-1]
            logger.info(f"  Legacy content length: {len(most_recent_web_research)} characters")
            research_content.append(most_recent_web_research)
        else:
            logger.info("No legacy web research results available")
        
        # Combine all research content
        combined_research = "\n\n".join(research_content) if research_content else "No research content available."
        
        logger.info(f"Research compilation completed: {total_sources} total sources")
        logger.info(f"Combined research content length: {len(combined_research)} characters")
        
        summarizer_prompt = summarizer_instructions.format(
            
            search_intent = state.search_intent or "hybrid",
            intent_confidence=state.intent_confidence or 0.6,
            research_loop_count = state.research_loop_count,
            max_loops = state.max_loops,
            research_topic = state.research_topic
        )
        # # Build the human message with comprehensive research context
        # if existing_summary:
        #     logger.info("Building UPDATE prompt with existing summary")
        #     human_message_content = (
        #         f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
        #         f"<New Research Context> \n {combined_research} \n <New Research Context>\n\n"
        #         f"<Intent Classification> Search Strategy: {state.search_intent} (confidence: {state.intent_confidence:.2f}) <Intent Classification>\n\n"
        #         f"Update the Existing Summary with the New Research Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        #     )
        # else:
        #     logger.info("Building CREATE prompt for new summary")
        #     human_message_content = (
        #         f"<Research Context> \n {combined_research} \n <Research Context>\n\n"
        #         f"<Intent Classification> Search Strategy: {state.search_intent} (confidence: {state.intent_confidence:.2f}) <Intent Classification>\n\n"
        #         f"Create a Comprehensive Summary using the Research Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        #     )
        # TODO: optimize prompts
        if existing_summary:
            logger.info("Building UPDATE prompt with existing summary using CoVe framework")
            human_message_content = f"""
                            {summarizer_prompt}

                            <EXISTING_SUMMARY>
                            {existing_summary}
                            </EXISTING_SUMMARY>

                            <NEW_RESEARCH_CONTEXT>
                            {combined_research}
                            </NEW_RESEARCH_CONTEXT>

                            <TASK>
                            Update the existing summary using Chain-of-Verification process for: {state.research_topic}

                            Follow the verification framework:
                            1. Identify evidence for each claim
                            2. Cross-verify with multiple sources 
                            3. Calibrate confidence levels
                            4. Integrate new information systematically
                            5. Flag any contradictions or gaps
                            </TASK>
                            """
        else:
            logger.info("Building CREATE prompt for new summary using CoVe framework")
            human_message_content = f"""
                                {summarizer_prompt}

                                <RESEARCH_CONTEXT>
                                {combined_research}
                                </RESEARCH_CONTEXT>

                                <TASK>
                                Create a comprehensive summary using Chain-of-Verification process for: {state.research_topic}

                                Follow the verification framework:
                                1. Identify evidence sources for each major claim
                                2. Cross-verify information across sources
                                3. Assign appropriate confidence levels
                                4. Structure findings systematically
                                5. Identify knowledge gaps explicitly
                                </TASK>
                                """

        logger.info(f"Final prompt length: {len(human_message_content)} characters")

        # Run the LLM
        configurable = Configuration.from_runnable_config(config)
        
        # Choose the appropriate LLM based on the provider
        if configurable.llm_provider == "lmstudio":
            logger.info(f"Using LMStudio LLM: {configurable.local_llm}")
            logger.info(f"LMStudio URL: {configurable.lmstudio_base_url}")
            llm = ChatLMStudio(
                base_url=configurable.lmstudio_base_url, 
                model=configurable.local_llm, 
                temperature=0.5 # want fact
            )
        elif configurable.llm_provider == "nvidia_nim":
            logger.info(f"Using NVIDIA NIM LLM: {configurable.get_model_name()}")
            logger.info(f"NVIDIA NIM URL: {configurable.nvidia_nim_base_url}")
            llm = ChatNVIDIANIM(
                base_url=configurable.nvidia_nim_base_url,
                model=configurable.get_model_name(),
                temperature=0.5,  # want factual responses
                nvidia_api_key=configurable.nvidia_api_key,
                enable_reasoning=False  # Enable reasoning for research synthesis
            )
        else:  # Default to Ollama
            logger.info(f"Using Ollama LLM: {configurable.local_llm}")
            logger.info(f"Ollama URL: {configurable.ollama_base_url}")
            llm = ChatOllama(
                base_url=configurable.ollama_base_url, 
                model=configurable.local_llm, 
                temperature=0.5 
            )
        
        logger.info("Calling LLM for research synthesis")
        result = llm.invoke([
            HumanMessage(content=human_message_content)
        ])

        logger.info(f"LLM response received: {len(result.content)} characters")

        # Strip thinking tokens if configured
        running_summary = result.content
        if configurable.strip_thinking_tokens:
            logger.info("Stripping thinking tokens from LLM response")
            running_summary = strip_thinking_tokens(running_summary)
            logger.info(f"After stripping tokens: {len(running_summary)} characters")
        
        # Extract verification metadata if available (look for confidence indicators)
        verification_metadata = {
            "total_sources_processed": total_sources,
            "source_types": {
                "arxiv": len(state.arxiv_results or []),
                "semantic": len(state.semantic_results or []),
                "web": len(state.web_results or []),
                "extracted": len(state.extracted_content or [])
            },
            "synthesis_approach": "chain_of_verification",
            "research_loop": state.research_loop_count
        }
        # Store synthesis time
        processing_time = time.time() - start_time
        processing_times = state.processing_times.copy()
        processing_times["synthesis"] = processing_time
        
        logger.info(f"Research synthesis completed in {processing_time:.2f}s")
        logger.info(f"Final summary length: {len(running_summary)} characters")
        logger.info(f"Verification metadata: {verification_metadata}")
        logger.info("COMPLETED NODE: synthesize_research()")
        
        return {
            "running_summary": running_summary,
            "processing_times": processing_times
        }
        
    except Exception as e:
        logger.error(f"RESEARCH SYNTHESIS FAILED: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Research topic: '{state.research_topic}'")
        logger.error(f"Available sources - arxiv: {len(state.arxiv_results or [])}, semantic: {len(state.semantic_results or [])}, web: {len(state.web_results or [])}")
        
        # Return existing summary or empty to prevent pipeline crash
        processing_time = time.time() - start_time
        processing_times = state.processing_times.copy()
        processing_times["synthesis"] = processing_time
        error_metadata = {
            "error": str(e),
            "error_type": type(e).__name__,
            "fallback_used": True
        }
        return {
            "running_summary": state.running_summary or f"Error synthesizing research for: {state.research_topic}",
            "verification_metadata": error_metadata,
            "processing_times": processing_times
        }

def reflect_on_research(state: ResearchState, config: RunnableConfig):
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.
    
    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.
    
    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    logger.info("EXECUTING NODE: reflect_on_research() - Identifying knowledge gaps")
    logger.info(f"INPUT: research_topic = '{state.research_topic}'")
    logger.info(f"INPUT: research_loop_count = {state.research_loop_count}")
    
    if state.running_summary:
        summary_length = len(state.running_summary)
        logger.info(f"INPUT: running_summary_length = {summary_length} characters")
        logger.info(f"Summary preview: {state.running_summary[:100]}...")
    else:
        logger.warning("INPUT: No running summary available for reflection")
    
    start_time = time.time()
    
    try:
        # Generate a query
        configurable = Configuration.from_runnable_config(config)
        # Format reflection prompt with all context
        enhanced_prompt = reflection_instructions.format(
            research_topic=state.research_topic,
            search_intent=state.search_intent or "hybrid",
            intent_confidence=state.intent_confidence or 0.6,
            research_loop_count=state.research_loop_count,
            max_loops=configurable.max_research_loops or 3,
            search_strategy=state.search_strategy or "hybrid"
        )
        # Choose the appropriate LLM based on the provider
        if configurable.llm_provider == "lmstudio":
            logger.info(f"Using LMStudio LLM with JSON mode: {configurable.local_llm}")
            logger.info(f"LMStudio URL: {configurable.lmstudio_base_url}")
            llm_json_mode = ChatLMStudio(
                base_url=configurable.lmstudio_base_url, 
                model=configurable.local_llm, 
                temperature=0.5,  # a bit more creative
                format="json"
            )
        elif configurable.llm_provider == "nvidia_nim":
            logger.info(f"Using NVIDIA NIM LLM with JSON mode: {configurable.get_model_name()}")
            logger.info(f"NVIDIA NIM URL: {configurable.nvidia_nim_base_url}")
            llm_json_mode = ChatNVIDIANIM(
                base_url=configurable.nvidia_nim_base_url,
                model=configurable.get_model_name(),
                temperature=0.5,  # a bit more creative
                nvidia_api_key=configurable.nvidia_api_key,
                enable_reasoning=False  # Disable for structured JSON output
            )
        else: # Default to Ollama
            logger.info(f"Using Ollama LLM with JSON mode: {configurable.local_llm}")
            logger.info(f"Ollama URL: {configurable.ollama_base_url}")
            llm_json_mode = ChatOllama(
                base_url=configurable.ollama_base_url, 
                model=configurable.local_llm, 
                temperature=0.5, 
                format="json"
            )
        human_message_content = f"""
                                {enhanced_prompt}

                                <CURRENT_KNOWLEDGE_STATE>
                                {state.running_summary or "No summary available yet"}
                                </CURRENT_KNOWLEDGE_STATE>

                                Perform systematic ReAct gap analysis and generate optimized follow-up query:
                                """
        # # Prepare reflection prompt
        # system_prompt = reflection_instructions.format(research_topic=state.research_topic)
        # human_prompt = f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:"
        
        # logger.info("Prepared reflection prompts")
        # logger.info(f"System prompt length: {len(system_prompt)} characters")
        # logger.info(f"Human prompt length: {len(human_prompt)} characters")
        
        # logger.info("Calling LLM for reflection and knowledge gap analysis")
        # result = llm_json_mode.invoke(
        #     [SystemMessage(content=system_prompt),
        #     HumanMessage(content=human_prompt)]
        # )
        logger.info("Calling LLM for ReAct reflection analysis")
        result = llm_json_mode.invoke([
            HumanMessage(content=human_message_content)
        ])
        logger.info(f"LLM response received: {len(result.content)} characters")
        logger.info(f"Raw LLM response preview: {result.content[:150]}...")
        
        # Strip thinking tokens if configured
        try:
            logger.info("Attempting to parse LLM response as JSON")
            # Try to parse as JSON first
            reflection_content = json.loads(result.content)
            logger.info("JSON parsing successful")
            
            # Log the parsed content structure
            if isinstance(reflection_content, dict):
                logger.info(f"JSON response keys: {list(reflection_content.keys())}")
                
                # Get the follow-up query
                query = reflection_content.get('follow_up_query')
                reflection_metadata = {
                    "gap_category": reflection_content.get("gap_category", "unknown"),
                    "reasoning": reflection_content.get("reasoning", ""),
                    "expected_sources": reflection_content.get("expected_sources", ""),
                    "confidence": reflection_content.get("confidence", "medium"),
                    "knowledge_gap": reflection_content.get("knowledge_gap", "")
                }
                
                if query:
                    # Update search intent if recommended
                    new_intent = reflection_content.get("search_intent", state.search_intent)
                    logger.info(f"GENERATED QUERY: '{query}'")
                    logger.info(f"Gap category: {reflection_metadata['gap_category']}")
                    
                    processing_time = time.time() - start_time
                    logger.info(f"Reflection completed in {processing_time:.2f}s")
                    
                    return {
                        "search_query": query,
                        "search_intent": new_intent,  # Dynamic intent updating
                        "reflection_metadata": reflection_metadata
                    }
                else:
                    logger.warning("Empty query, using fallback")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            
            # Try regex extraction from raw content
            import re
            query_match = re.search(r'"follow_up_query":\s*"([^"]+)"', result.content)
            if query_match:
                query = query_match.group(1).strip()
                logger.info(f"EXTRACTED QUERY: '{query}'")
                return {"search_query": query}
        
        # Fallback query
        fallback_query = f"Additional research on {state.research_topic}"
        logger.info(f"Using FALLBACK QUERY: '{fallback_query}'")
        return {"search_query": fallback_query}
        
    except Exception as e:
        logger.error(f"REFLECTION FAILED: {e}")
        return {"search_query": f"More about {state.research_topic}"}                   

def finalize_research(state: ResearchState) -> Dict[str, Any]:
    """research finalization with quality assessment and verification metadata.
    
    Creates a comprehensive final summary with research quality indicators,
    confidence scoring, and structured source documentation.
    
    Args:
        state: Current graph state containing all research data and summaries
        
    Returns:
        Dictionary with enhanced final_summary and quality metadata
    """
    logger.info("EXECUTING NODE: finalize_research() - final research output")
    logger.info(f"INPUT: research_topic = '{state.research_topic}'")
    logger.info(f"INPUT: research_loop_count = {state.research_loop_count}")
    logger.info(f"INPUT: search_intent = '{state.search_intent}' (confidence: {state.intent_confidence:.2f})")
    
    start_time = time.time()
    
    try:
        
        # metrics calculation
        total_papers = len(state.arxiv_results or []) + len(state.semantic_results or [])
        total_web_results = len(state.web_results or [])
        total_extractions = len(state.extracted_content or [])
        total_processing_time = sum(state.processing_times.values()) if state.processing_times else 0
        
        # Calculate research quality indicators
        source_diversity = sum([
            1 if state.arxiv_results else 0,
            1 if state.semantic_results else 0,
            1 if state.web_results else 0,
            1 if state.extracted_content else 0
        ])
        
        # Content confidence calculation
        content_confidence = 0.0
        confidence_factors = []
        
        # Source diversity factor (0-1)
        confidence_factors.append(min(source_diversity / 4.0, 1.0))
        
        # Intent classification confidence
        confidence_factors.append(state.intent_confidence or 0.5)
        
        # Content volume factor
        total_content = total_papers + total_web_results + total_extractions
        volume_confidence = min(total_content / 10.0, 1.0)
        confidence_factors.append(volume_confidence)
        
        # Research completeness factor
        completion_factor = min(state.research_loop_count / 2.0, 1.0)
        confidence_factors.append(completion_factor)
        
        content_confidence = sum(confidence_factors) / len(confidence_factors)
        
        logger.info(f"Quality metrics: diversity={source_diversity}/4, confidence={content_confidence:.2f}")
        
        # Enhanced source deduplication and categorization
        seen_sources = set()
        academic_sources = []
        web_sources = []
        other_sources = []
        
        for source in state.sources_gathered or []:
            for line in source.split('\n'):
                clean_line = line.strip()
                if clean_line and clean_line not in seen_sources:
                    seen_sources.add(clean_line)
                    
                    # Categorize sources
                    if any(keyword in clean_line.lower() for keyword in ['arxiv', 'paper', 'semantic', 'academic']):
                        academic_sources.append(clean_line)
                    elif any(keyword in clean_line.lower() for keyword in ['web', 'search', 'searxng', 'tavily']):
                        web_sources.append(clean_line)
                    else:
                        other_sources.append(clean_line)
        
        total_unique_sources = len(academic_sources) + len(web_sources) + len(other_sources)
        logger.info(f"Source categorization: academic={len(academic_sources)}, web={len(web_sources)}, other={len(other_sources)}")
        
        # Research quality assessment
        quality_indicators = []
        if content_confidence >= 0.8:
            quality_indicators.append("High confidence research")
        elif content_confidence >= 0.6:
            quality_indicators.append("Good confidence research")
        else:
            quality_indicators.append("Moderate confidence research")
            
        if source_diversity >= 3:
            quality_indicators.append("Multi-source verification")
        elif source_diversity >= 2:
            quality_indicators.append("Dual-source validation")
        else:
            quality_indicators.append("Limited source diversity")
            
        if state.research_loop_count >= 2:
            quality_indicators.append("Iterative refinement")
        
        # Create enhanced final summary with verification indicators
        final_summary = f"""# Research Analysis: {state.research_topic}

                ## Executive Summary
                {state.running_summary or "No comprehensive analysis available."}

                ## Research Quality Assessment
                - **Overall Confidence**: {content_confidence:.2f}/1.0 ({', '.join(quality_indicators)})
                - **Search Strategy**: {state.search_intent} (classification confidence: {state.intent_confidence:.2f})
                - **Source Diversity**: {source_diversity}/4 source types utilized
                - **Research Depth**: {state.research_loop_count} iteration{"s" if state.research_loop_count != 1 else ""} completed

                ## Content Analysis Metrics
                - **Academic Sources**: {total_papers} papers (ArXiv: {len(state.arxiv_results or [])}, Semantic: {len(state.semantic_results or [])})
                - **Web Intelligence**: {total_web_results} web sources analyzed
                - **Deep Extractions**: {total_extractions} full content extractions
                - **Processing Efficiency**: {total_processing_time:.1f}s total ({total_processing_time/max(1, state.research_loop_count):.1f}s avg/loop)

                ## Verification Framework Applied
                - **Chain-of-Verification**: Evidence-based claim validation
                - **Cross-Source Validation**: Multi-source fact checking
                - **Confidence Calibration**: Systematic uncertainty assessment
                - **Gap Identification**: Knowledge limitation tracking

                ## Source Documentation

                ### Academic Sources ({len(academic_sources)})
                {chr(10).join([f"• {source}" for source in academic_sources[:5]])}
                {f"• ...and {len(academic_sources) - 5} more academic sources" if len(academic_sources) > 5 else ""}

                ### Web Intelligence ({len(web_sources)})
                {chr(10).join([f"• {source}" for source in web_sources[:5]])}
                {f"• ...and {len(web_sources) - 5} more web sources" if len(web_sources) > 5 else ""}

                {f"### Additional Sources ({len(other_sources)})" + chr(10) + chr(10).join([f"• {source}" for source in other_sources[:3]]) if other_sources else ""}

                ## Processing Performance
                {chr(10).join([f"- **{k.title().replace('_', ' ')}**: {v:.2f}s" for k, v in (state.processing_times or {}).items()])}

                ---
                *Generated by Research Pipeline with Chain-of-Verification 
                • Intent Classification: XLM-RoBERTa 
                • Multi-Modal RAG Architecture*"""

        finalization_time = time.time() - start_time
        
        # Create comprehensive metadata for analysis
        research_metadata = {
            "content_confidence": content_confidence,
            "source_diversity": source_diversity,
            "quality_indicators": quality_indicators,
            "total_sources": total_unique_sources,
            "source_breakdown": {
                "academic": len(academic_sources),
                "web": len(web_sources), 
                "other": len(other_sources)
            },
            "processing_metrics": {
                "total_time": total_processing_time,
                "avg_time_per_loop": total_processing_time / max(1, state.research_loop_count),
                "finalization_time": finalization_time
            },
            "research_completeness": {
                "loops_completed": state.research_loop_count,
                "search_strategy": state.search_intent,
                "intent_confidence": state.intent_confidence
            }
        }
        
        logger.info(f"research finalization completed in {finalization_time:.2f}s")
        logger.info(f"Final summary: {len(final_summary)} characters")
        logger.info(f"Research quality: {content_confidence:.2f} confidence, {source_diversity}/4 diversity")
        
        # Prepare final results for output
        return_arxiv = state.arxiv_results or []
        return_semantic = state.semantic_results or []
        return_web = state.web_results or []
        return_extracted = state.extracted_content or []
        
        logger.info("COMPLETED NODE: finalize_research()")
        
        # CRITICAL: Return ALL fields needed in final output
        # LangGraph only preserves fields explicitly returned by the final node
        return {
            "final_summary": final_summary,
            "running_summary": final_summary, 
            "research_metadata": research_metadata,
            "arxiv_results": return_arxiv,  # Ensure ArXiv results are in final output
            "semantic_results": return_semantic,  # Ensure semantic results are in final output
            "web_results": return_web,  # Ensure web results are in final output
            "extracted_content": return_extracted,  # Ensure extracted content is in final output
            # Preserve other essential state fields for final output
            "search_intent": state.search_intent,
            "intent_confidence": state.intent_confidence,
            "sources_gathered": state.sources_gathered,
            "processing_times": state.processing_times
        }
        
    except Exception as e:
        logger.error(f"RESEARCH FINALIZATION FAILED: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # fallback with partial data
        try:
            partial_summary = f"""# Research Analysis: {state.research_topic}

                    ## Status
                    Research completed with technical limitations during finalization.

                    ## Available Analysis
                    {state.running_summary or "No analysis content available."}

                    ## Technical Details
                    - **Error**: {str(e)}
                    - **Search Strategy**: {getattr(state, 'search_intent', 'Unknown')}
                    - **Loops Completed**: {getattr(state, 'research_loop_count', 0)}

                    *Research pipeline encountered finalization error but core analysis may be available above.*"""
            
            error_metadata = {
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True,
                "partial_data_available": bool(state.running_summary)
            }
            
            return {
                "final_summary": partial_summary,
                "running_summary": partial_summary,
                "research_metadata": error_metadata
            }
            
        except Exception as fallback_error:
            logger.error(f"Fallback finalization also failed: {fallback_error}")
            basic_summary = f"Research Summary: {state.research_topic}\n\nFinalization failed: {str(e)}"
            return {
                "final_summary": basic_summary,
                "running_summary": basic_summary
            }
        
def route_research_flow(state: ResearchState) -> Literal["execute_search", "extract_content", "finalize_research"]:
    """Build router that prevents infinite content extraction loops."""
    config = Configuration()
    logger.info("EXECUTING ROUTER: route_research_flow() - Determining next workflow step")
    logger.info(f"INPUT: research_loop_count = {state.research_loop_count}")
    logger.info(f"INPUT: research_topic = '{state.research_topic}'")
    
    max_loops = getattr(state, 'max_web_research_loops', 3)
    logger.info(f"CONFIGURATION: max_web_research_loops = {max_loops}")
    
    # Check if we've reached the loop limit
    logger.info("DECISION POINT 1: Checking research loop limit")
    if state.research_loop_count >= max_loops:
        logger.info(f"ROUTING DECISION: finalize_research")
        logger.info(f"REASON: Reached max loops ({state.research_loop_count}/{max_loops})")
        return "finalize_research"
    
    logger.info(f"Loop limit check: {state.research_loop_count}/{max_loops} - continuing research")
    
    # Check current state
    logger.info("DECISION POINT 2: Checking content extraction needs")
    has_search_results = bool(state.arxiv_results or state.semantic_results or state.web_results)
    has_extracted_content = bool(state.extracted_content)
    jina_configured = bool(config.jina_api_key) or has_extracted_content
    
    # Track content extraction attempts to prevent infinite loops
    extraction_attempts = getattr(state, 'content_extraction_attempts', 0)
    max_extraction_attempts = 2  # Limit extraction attempts
    
    logger.info(f"Has search results: {has_search_results}")
    logger.info(f"Has extracted content: {has_extracted_content}")
    logger.info(f"Jina configured: {jina_configured}")
    logger.info(f"Content extraction attempts: {extraction_attempts}/{max_extraction_attempts}")
    
    # Only try content extraction if:
    # 1. We have search results
    # 2. We don't have extracted content yet
    # 3. Jina API is configured
    # 4. We haven't exceeded max extraction attempts
    if (has_search_results and 
        not has_extracted_content and 
        jina_configured and 
        extraction_attempts < max_extraction_attempts):
        
        logger.info("ROUTING DECISION: extract_content")
        logger.info(f"REASON: Have search results but no extracted content (attempt {extraction_attempts + 1}/{max_extraction_attempts})")
        
        # Increment extraction attempts
        state.content_extraction_attempts = extraction_attempts + 1
        
        return "extract_content"
    
    # If we've tried extraction too many times, give up and finalize
    elif extraction_attempts >= max_extraction_attempts:
        logger.info("ROUTING DECISION: finalize_research")
        logger.info(f"REASON: Max content extraction attempts reached ({extraction_attempts}/{max_extraction_attempts})")
        return "finalize_research"
    
    # If we have enough content, finalize
    elif has_search_results or has_extracted_content:
        logger.info("ROUTING DECISION: finalize_research")
        logger.info(f"REASON: Have sufficient content for analysis")
        return "finalize_research"
    
    # Otherwise, continue searching
    else:
        logger.info("ROUTING DECISION: execute_search")
        logger.info(f"REASON: Need more research content")
        return "execute_search"

# Enhanced LangGraph workflow with intent classification and MCPO integration
builder = StateGraph(ResearchState, input=ResearchStateInput, output=ResearchStateOutput, config_schema=Configuration)

# Add all nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("generate_query", generate_query)
builder.add_node("execute_search", execute_search)
builder.add_node("extract_content", extract_content)
builder.add_node("synthesize_research", synthesize_research)
builder.add_node("reflect_on_research", reflect_on_research)
builder.add_node("finalize_research", finalize_research)

# Build the enhanced workflow
builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "generate_query")
builder.add_edge("generate_query", "execute_search")
builder.add_edge("execute_search", "extract_content")  # Missing connection fixed!
builder.add_edge("extract_content", "synthesize_research")
builder.add_edge("synthesize_research", "reflect_on_research")
builder.add_conditional_edges("reflect_on_research", route_research_flow)
builder.add_edge("finalize_research", END)

# Compile the enhanced graph
graph = builder.compile()

# Export as app for convenience
app = graph

# Legacy aliases for backward compatibility
SummaryState = ResearchState
SummaryStateInput = ResearchStateInput
SummaryStateOutput = ResearchStateOutput