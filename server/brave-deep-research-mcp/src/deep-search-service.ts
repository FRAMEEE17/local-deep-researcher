import { BraveSearchClient, BraveSearchResult } from './brave-search-client.js';
import { ContentExtractor, ExtractedContent } from './content-extractor.js';
import { logger } from './utils/logger.js';

/**
 * Parameters for deep search
 */
export interface DeepSearchParams {
  query: string;
  count?: number;
  depth?: number;
}

/**
 * Search result with extracted content
 */
export interface EnrichedSearchResult {
  url: string;
  title: string;
  description: string;
  extractedContent?: ExtractedContent;
  relatedPages?: ExtractedContent[];
}

/**
 * Deep search response
 */
export interface DeepSearchResponse {
  query: string;
  results: EnrichedSearchResult[];
  summary: string;
}

/**
 * Service that combines Brave Search with content extraction
 */
export class DeepSearchService {
  private braveSearchClient: BraveSearchClient;
  
  /**
   * Constructor
   * @param apiKey The Brave Search API key
   */
  constructor(apiKey: string) {
    this.braveSearchClient = new BraveSearchClient(apiKey);
  }
  
  /**
   * Perform a deep search
   * @param params The search parameters
   * @returns The search results with extracted content
   */
  public async deepSearch(params: DeepSearchParams): Promise<DeepSearchResponse> {
    const { query, count = 5, depth = 0 } = params;
    
    // Enforce reasonable limits
    const actualCount = Math.min(count, parseInt(process.env.MAX_PAGES_PER_SEARCH || '5', 10));
    const actualDepth = Math.min(depth, 2); // Maximum depth of 2 to avoid excessive crawling
    
    logger.info(`Starting deep search for "${query}" (count: ${actualCount}, depth: ${actualDepth})`);
    
    try {
      // Perform the initial search
      const searchResults = await this.braveSearchClient.webSearch({
        query,
        count: actualCount
      });
      
      // Process each search result
      const enrichedResults = await this.processSearchResults(searchResults, actualDepth);
      
      // Generate a summary of the results
      const summary = this.generateSummary(query, enrichedResults);
      
      return {
        query,
        results: enrichedResults,
        summary
      };
    } catch (error) {
      logger.error(`Deep search error for "${query}":`, error);
      throw error;
    }
  }
  
  /**
   * Process search results by extracting content and related pages
   * @param searchResults The initial search results
   * @param depth The depth of related pages to extract
   * @returns The enriched search results
   */
  private async processSearchResults(
    searchResults: BraveSearchResult[], 
    depth: number
  ): Promise<EnrichedSearchResult[]> {
    const enrichedResults: EnrichedSearchResult[] = [];
    
    // Process each search result sequentially to avoid overwhelming resources
    for (const result of searchResults) {
      try {
        logger.debug(`Processing search result: ${result.url}`);
        
        // Extract content from the main page
        const extractedContent = await ContentExtractor.extractContent(result.url);
        
        const enrichedResult: EnrichedSearchResult = {
          ...result,
          extractedContent
        };
        
        // If depth > 0, process related pages
        if (depth > 0 && extractedContent.links && extractedContent.links.length > 0) {
          enrichedResult.relatedPages = await this.processRelatedPages(
            extractedContent.links,
            depth
          );
        }
        
        enrichedResults.push(enrichedResult);
      } catch (error) {
        logger.error(`Error processing search result ${result.url}:`, error);
        // Still add the result, but without the extracted content
        enrichedResults.push(result);
      }
    }
    
    return enrichedResults;
  }
  
  /**
   * Process related pages by extracting their content
   * @param links The links to process
   * @param depth The remaining depth to process
   * @returns The extracted content of related pages
   */
  private async processRelatedPages(
    links: { url: string; text: string }[],
    depth: number
  ): Promise<ExtractedContent[]> {
    // Limit the number of related pages to process
    const pagesToProcess = links.slice(0, 3);
    const relatedPages: ExtractedContent[] = [];
    
    // Process each link sequentially
    for (const link of pagesToProcess) {
      try {
        logger.debug(`Processing related page: ${link.url}`);
        
        // Extract content from the related page
        const extractedContent = await ContentExtractor.extractContent(link.url);
        relatedPages.push(extractedContent);
        
        // Continue recursively if depth > 1
        if (depth > 1 && extractedContent.links && extractedContent.links.length > 0) {
          // Only process a single link from each related page for depth > 1
          const nextLevelLink = extractedContent.links[0];
          logger.debug(`Processing nested related page: ${nextLevelLink.url}`);
          
          const nestedContent = await ContentExtractor.extractContent(nextLevelLink.url);
          relatedPages.push(nestedContent);
        }
      } catch (error) {
        logger.error(`Error processing related page ${link.url}:`, error);
        // Continue with the next link
      }
    }
    
    return relatedPages;
  }
  
  /**
   * Generate a summary of the search results
   * @param query The original search query
   * @param results The enriched search results
   * @returns A summary of the results
   */
  private generateSummary(query: string, results: EnrichedSearchResult[]): string {
    const validResults = results.filter(r => r.extractedContent && !r.extractedContent.error);
    
    if (validResults.length === 0) {
      return `No valid results found for "${query}".`;
    }
    
    const totalMainPages = results.length;
    const successfulMainPages = validResults.length;
    
    let totalRelatedPages = 0;
    let successfulRelatedPages = 0;
    
    for (const result of validResults) {
      if (result.relatedPages) {
        totalRelatedPages += result.relatedPages.length;
        successfulRelatedPages += result.relatedPages.filter(p => !p.error).length;
      }
    }
    
    return `Deep search for "${query}" found ${totalMainPages} results, with content successfully extracted from ${successfulMainPages} pages. Additionally, ${totalRelatedPages} related pages were analyzed, with ${successfulRelatedPages} successfully processed.`;
  }
}
