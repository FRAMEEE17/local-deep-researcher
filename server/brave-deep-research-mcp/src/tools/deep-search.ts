import { searchWithBrave } from '../services/brave-search.js';
import { performDeepSearch } from '../services/puppeteer.js';
import { debug } from '../utils/config.js';

/**
 * Deep search tool implementation for MCP
 */
export const deepSearchTool = {
  name: "deep-search",
  description: "Perform a deep web search that visits pages to extract full content",
  
  // Define input schema
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Search query" },
      results: { type: "number", description: "Number of search results to process (default: 3, max: 10)" },
      depth: { type: "number", description: "Depth of link traversal for each result (default: 1, max: 3)" },
    },
    required: ["query"]
  },
  
  // Tool execution handler
  async handler(params: any) {
    const { query, results = 3, depth = 1 } = params;
    
    // Validate and cap parameters
    const maxResults = Math.min(Math.max(1, parseInt(String(results)) || 3), 10);
    const maxDepth = Math.min(Math.max(1, parseInt(String(depth)) || 1), 3);
    
    debug(`Executing deep-search with query: ${query}, results: ${maxResults}, depth: ${maxDepth}`);
    
    try {
      // Step 1: Get initial search results from Brave Search API
      const searchResponse = await searchWithBrave({
        q: query,
        count: maxResults
      });
      
      if (searchResponse.results.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `No search results found for query: "${query}"`
            }
          ]
        };
      }
      
      // Step 2: Extract URLs from search results
      const urls = searchResponse.results.map(result => result.url);
      
      // Step 3: Perform deep search on the URLs
      const deepSearchResults = await performDeepSearch(urls, {
        depth: maxDepth,
        maxPages: maxResults * maxDepth
      });
      
      // Step 4: Format the results
      const formattedResults = deepSearchResults.map(result => {
        return `
# ${result.title}
URL: ${result.url}
${result.description ? `Description: ${result.description}` : ''}

## Content
${result.content.slice(0, 1000)}${result.content.length > 1000 ? '...' : ''}
        `.trim();
      });
      
      // Generate summary
      const summary = `
# Deep Search Results for "${query}"
Found ${deepSearchResults.length} pages with depth ${maxDepth}

${formattedResults.join('\n\n---\n\n')}
      `.trim();
      
      return {
        content: [
          {
            type: "text",
            text: summary
          }
        ]
      };
    } catch (error) {
      debug('Deep search error:', error);
      
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Error performing deep search: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }
};