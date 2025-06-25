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
# Try to import intent classifier, fallback if dependencies missing
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
    start_time = time.time()
    
    configurable = Configuration.from_runnable_config(config)
    
    if INTENT_CLASSIFIER_AVAILABLE:
        # Use trained XLM-RoBERTa model
        intent_result = classify_query_intent(state.research_topic, configurable)
        
        return {
            "search_intent": intent_result["routing_strategy"],
            "intent_confidence": intent_result["confidence"],
            "search_strategy": intent_result["routing_strategy"],
            "processing_times": {"intent_classification": time.time() - start_time}
        }
    else:
        # Fallback: simple rule-based classification
        topic_lower = state.research_topic.lower()
        
        # Rule-based intent detection
        if any(word in topic_lower for word in ['arxiv', 'paper', 'research', 'academic', 'study', 'journal']):
            strategy = "arxiv_search"
            confidence = 0.8
        elif any(word in topic_lower for word in ['news', 'current', 'latest', 'recent', 'today']):
            strategy = "web_search"
            confidence = 0.7
        else:
            strategy = "hybrid_search"
            confidence = 0.6
        
        return {
            "search_intent": strategy,
            "intent_confidence": confidence,
            "search_strategy": strategy,
            "processing_times": {"intent_classification": time.time() - start_time}
        }

def generate_query(state: ResearchState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.
    
    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    start_time = time.time()
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    
    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        query = json.loads(content)
        search_query = query['query']
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = content
    
    # Store the generation time
    processing_times = state.processing_times.copy()
    processing_times["query_generation"] = time.time() - start_time
    
    return {
        "search_query": search_query,
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
    start_time = time.time()
    
    configurable = Configuration.from_runnable_config(config)
    
    if SEARCH_ENGINES_AVAILABLE:
        search_engines = create_search_engines(configurable)
        
        # Execute search based on intent routing
        search_result = await search_engines.execute_search_strategy(
            query=state.search_query,
            strategy=state.search_intent,
            max_results=configurable.max_papers_per_search
        )
    else:
        # Fallback to legacy search methods
        return await legacy_web_research(state, config)
    
    processing_time = time.time() - start_time
    
    # Update processing times
    processing_times = state.processing_times.copy()
    processing_times["search_execution"] = processing_time
    
    # Route results based on search strategy
    if state.search_intent == "arxiv_search":
        return {
            "arxiv_results": [{"instruction": search_result.get("instruction"), "query": state.search_query}],
            "sources_gathered": [f"ArXiv MCP search for: {state.search_query}"],
            "processing_times": processing_times,
            "research_loop_count": state.research_loop_count + 1
        }
    elif state.search_intent == "web_search":
        return {
            "web_results": search_result.get("results", []),
            "sources_gathered": [f"SearXNG web search: {len(search_result.get('results', []))} results"],
            "processing_times": processing_times,
            "research_loop_count": state.research_loop_count + 1
        }
    elif state.search_intent == "hybrid_search":
        return {
            "semantic_results": search_result.get("papers", []),
            "arxiv_results": [{"instruction": search_result.get("instruction"), "query": state.search_query}],
            "sources_gathered": [f"Hybrid search: {len(search_result.get('papers', []))} semantic + ArXiv MCP"],
            "processing_times": processing_times,
            "research_loop_count": state.research_loop_count + 1
        }
    else:
        # Fallback to legacy web research
        return await legacy_web_research(state, config)

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

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, fetch_full_page=configurable.fetch_full_page, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "searxng":
        search_results = searxng_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "arxiv":
        search_results = await arxiv_search(state.search_query, max_results=5)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1500, fetch_full_page=True)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)], 
        "research_loop_count": state.research_loop_count + 1, 
        "web_research_results": [search_str]
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
    start_time = time.time()
    
    configurable = Configuration.from_runnable_config(config)
    
    if not SEARCH_ENGINES_AVAILABLE:
        # Skip content extraction if search engines not available
        processing_time = time.time() - start_time
        processing_times = state.processing_times.copy()
        processing_times["content_extraction"] = processing_time
        
        return {
            "extracted_content": [],
            "jina_extractions": [],
            "processing_times": processing_times
        }
    
    search_engines = create_search_engines(configurable)
    extracted_content = []
    
    # Extract from semantic results (ArXiv papers)
    if state.semantic_results:
        top_papers = state.semantic_results[:configurable.max_content_extractions]
        
        extraction_tasks = []
        for paper in top_papers:
            if paper.get('pdf_url'):
                extraction_tasks.append(
                    search_engines.extract_content_jina(paper['pdf_url'])
                )
        
        if extraction_tasks:
            extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            for result in extraction_results:
                if isinstance(result, dict) and result.get('success'):
                    extracted_content.append(result)
    
    # Extract from web results
    if state.web_results:
        top_web = state.web_results[:2]  # Limit web extractions
        
        for result in top_web:
            if result.get('url'):
                extraction = await asyncio.to_thread(
                    search_engines.extract_content_jina, result['url']
                )
                if extraction.get('success'):
                    extracted_content.append(extraction)
    
    processing_time = time.time() - start_time
    processing_times = state.processing_times.copy()
    processing_times["content_extraction"] = processing_time
    
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
    start_time = time.time()
    
    # Existing summary
    existing_summary = state.running_summary

    # Compile all research sources
    research_content = []
    
    # Add ArXiv results
    if state.arxiv_results:
        arxiv_summary = f"ArXiv Results ({len(state.arxiv_results)} found):\n"
        for result in state.arxiv_results:
            if 'instruction' in result:
                arxiv_summary += f"- MCPO instruction: {result['instruction']}\n"
            else:
                arxiv_summary += f"- {result.get('title', 'Unknown title')}\n"
        research_content.append(arxiv_summary)
    
    # Add semantic search results
    if state.semantic_results:
        semantic_summary = f"Semantic ArXiv Results ({len(state.semantic_results)} found):\n"
        for paper in state.semantic_results[:3]:  # Top 3
            semantic_summary += f"- {paper.get('title', 'Unknown')}: {paper.get('abstract', '')[:200]}...\n"
        research_content.append(semantic_summary)
    
    # Add web results
    if state.web_results:
        web_summary = f"Web Search Results ({len(state.web_results)} found):\n"
        for result in state.web_results[:3]:  # Top 3
            web_summary += f"- {result.get('title', 'Unknown')}: {result.get('content', '')[:200]}...\n"
        research_content.append(web_summary)
    
    # Add extracted content
    if state.extracted_content:
        content_summary = f"Extracted Content ({len(state.extracted_content)} sources):\n"
        for content in state.extracted_content[:2]:  # Top 2
            content_summary += f"- Content length: {content.get('length', 0)} chars\n"
        research_content.append(content_summary)
    
    # Fallback to legacy web research results
    if hasattr(state, 'web_research_results') and state.web_research_results:
        most_recent_web_research = state.web_research_results[-1]
        research_content.append(most_recent_web_research)
    
    # Combine all research content
    combined_research = "\n\n".join(research_content) if research_content else "No research content available."

    # Build the human message with comprehensive research context
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Research Context> \n {combined_research} \n <New Research Context>\n\n"
            f"<Intent Classification> Search Strategy: {state.search_intent} (confidence: {state.intent_confidence:.2f}) <Intent Classification>\n\n"
            f"Update the Existing Summary with the New Research Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Research Context> \n {combined_research} \n <Research Context>\n\n"
            f"<Intent Classification> Search Strategy: {state.search_intent} (confidence: {state.intent_confidence:.2f}) <Intent Classification>\n\n"
            f"Create a Comprehensive Summary using the Research Context on this topic: \n <User Input> \n {state.research_topic} \n <User Input>\n\n"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0
        )
    
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    # Strip thinking tokens if configured
    running_summary = result.content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)

    # Store synthesis time
    processing_times = state.processing_times.copy()
    processing_times["synthesis"] = time.time() - start_time
    
    return {
        "running_summary": running_summary,
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

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:")]
    )
    
    # Strip thinking tokens if configured
    try:
        # Try to parse as JSON first
        reflection_content = json.loads(result.content)
        # Get the follow-up query
        query = reflection_content.get('follow_up_query')
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            return {"search_query": f"Tell me more about {state.research_topic}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"search_query": f"Tell me more about {state.research_topic}"}
        
def finalize_research(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node that finalizes the research summary.
    
    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.
    
    Args:
        state: Current graph state containing the running summary and sources gathered
        
    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """

    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []
    
    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split('\n'):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)
    
    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    # Create comprehensive final summary with metrics
    total_papers = len(state.arxiv_results) + len(state.semantic_results)
    total_web_results = len(state.web_results)
    total_extractions = len(state.extracted_content)
    
    final_summary = f"""## Research Summary: {state.research_topic}

**Intent Classification:** {state.search_intent} (confidence: {state.intent_confidence:.2f})
**Search Strategy:** {state.search_strategy}

{state.running_summary}

### Research Metrics
- Total Papers Found: {total_papers}
- Web Results: {total_web_results}  
- Content Extractions: {total_extractions}
- Research Loops: {state.research_loop_count}

### Processing Times
{chr(10).join([f"- {k.title().replace('_', ' ')}: {v:.2f}s" for k, v in state.processing_times.items()])}

### Sources
{all_sources}
"""
    
    return {
        "final_summary": final_summary,
        "running_summary": final_summary
    }

def route_research_flow(state: ResearchState, config: RunnableConfig) -> Literal["finalize_research", "extract_content", "execute_search"]:
    """LangGraph routing function that determines the next step in the research flow.
    
    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.
    
    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting
        
    Returns:
        String literal indicating the next node to visit based on research progress
    """

    configurable = Configuration.from_runnable_config(config)
    
    # Check if we've reached the maximum research loops
    if state.research_loop_count > configurable.max_web_research_loops:
        return "finalize_research"
    
    # If we have search results but no extracted content, extract content
    has_results = (state.arxiv_results or state.semantic_results or state.web_results)
    needs_extraction = has_results and not state.extracted_content
    
    if needs_extraction and configurable.jina_api_key:
        return "extract_content"
    
    # Continue with more searches if needed
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
builder.add_edge("execute_search", "synthesize_research")
builder.add_edge("synthesize_research", "reflect_on_research")
builder.add_conditional_edges("reflect_on_research", route_research_flow)
builder.add_edge("extract_content", "synthesize_research")
builder.add_edge("finalize_research", END)

# Compile the enhanced graph
graph = builder.compile()

# Legacy aliases for backward compatibility
SummaryState = ResearchState
SummaryStateInput = ResearchStateInput
SummaryStateOutput = ResearchStateOutput