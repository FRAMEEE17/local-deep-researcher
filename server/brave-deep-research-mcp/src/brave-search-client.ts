import fetch from 'node-fetch';
import { logger } from './utils/logger.js';

/**
 * Interface for Brave Search API response
 */
export interface BraveSearchResult {
  url: string;
  title: string;
  description: string;
}

/**
 * Interface for web search parameters
 */
export interface WebSearchParams {
  query: string;
  count?: number;
  offset?: number;
}

/**
 * Class to handle interactions with the Brave Search API
 */
export class BraveSearchClient {
  private readonly apiKey: string;
  private readonly baseUrl = 'https://api.search.brave.com/res/v1/web/search';
  
  /**
   * Constructor
   * @param apiKey The Brave Search API key
   */
  constructor(apiKey: string) {
    if (!apiKey) {
      throw new Error('Brave Search API key is required');
    }
    this.apiKey = apiKey;
  }
  
  /**
   * Perform a web search using the Brave Search API
   * @param params The search parameters
   * @returns The search results
   */
  public async webSearch(params: WebSearchParams): Promise<BraveSearchResult[]> {
    const { query, count = 10, offset = 0 } = params;
    
    if (!query) {
      throw new Error('Search query is required');
    }
    
    const url = new URL(this.baseUrl);
    url.searchParams.append('q', query);
    url.searchParams.append('count', count.toString());
    url.searchParams.append('offset', offset.toString());
    
    try {
      logger.debug(`Performing Brave web search: ${query}`);
      
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-Subscription-Token': this.apiKey
        }
      });
      
      if (!response.ok) {
        throw new Error(`Brave Search API error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json() as any;
      
      if (!data.web || !data.web.results) {
        return [];
      }
      
      // Transform results to the expected format
      return data.web.results.map((result: any) => ({
        url: result.url,
        title: result.title,
        description: result.description
      }));
    } catch (error) {
      logger.error('Brave Search API error:', error);
      throw error;
    }
  }
}
