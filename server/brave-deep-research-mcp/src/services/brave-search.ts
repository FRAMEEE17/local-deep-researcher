import fetch from 'node-fetch';
import { config, debug } from '../utils/config.js';

const BRAVE_SEARCH_API = 'https://api.search.brave.com/res/v1/web/search';

export interface BraveSearchParams {
  q: string;
  count?: number;
  offset?: number;
}

export interface BraveSearchResult {
  title: string;
  url: string;
  description: string;
}

export interface BraveSearchResponse {
  results: BraveSearchResult[];
  total?: number;
  query?: string;
}

/**
 * Perform a search using the Brave Search API
 */
export async function searchWithBrave(params: BraveSearchParams): Promise<BraveSearchResponse> {
  debug(`Searching Brave for: ${params.q}`);
  
  const url = new URL(BRAVE_SEARCH_API);
  url.searchParams.append('q', params.q);
  
  if (params.count) {
    url.searchParams.append('count', params.count.toString());
  }
  
  if (params.offset) {
    url.searchParams.append('offset', params.offset.toString());
  }
  
  try {
    const response = await fetch(url.toString(), {
      headers: {
        'Accept': 'application/json',
        'X-Subscription-Token': config.BRAVE_API_KEY
      }
    });
    
    if (!response.ok) {
      throw new Error(`Brave Search API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json() as any;
    
    // Transform the Brave Search API response to our format
    const results: BraveSearchResult[] = data.web?.results?.map((result: any) => ({
      title: result.title || '',
      url: result.url || '',
      description: result.description || ''
    })) || [];
    
    return {
      results,
      total: data.web?.totalResults || 0,
      query: data.query?.rawQuery || params.q
    };
  } catch (error) {
    debug('Brave Search API error:', error);
    throw error;
  }
}