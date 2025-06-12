import { getBrowser } from '../utils/browser.js';
import { extractMainContent, extractLinks, extractMetadata } from '../utils/content-extractor.js';
import { config, debug } from '../utils/config.js';

export interface PageContent {
  url: string;
  title: string;
  description: string;
  content: string;
  links: Array<{ url: string, text: string }>;
}

export interface DeepSearchOptions {
  depth?: number;
  maxPages?: number;
}

/**
 * Extract content from a URL using Puppeteer
 */
export async function extractContentFromUrl(url: string): Promise<PageContent> {
  debug(`Extracting content from URL: ${url}`);
  
  const browser = await getBrowser();
  const page = await browser.newPage();
  
  try {
    // Set default timeout
    page.setDefaultTimeout(config.PAGE_TIMEOUT);
    
    // Set user agent to mimic a real browser
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');
    
    // Navigate to the URL
    await page.goto(url, { waitUntil: 'networkidle2' });
    
    // Extract metadata
    const metadata = await extractMetadata(page);
    
    // Extract main content
    const content = await extractMainContent(page);
    
    // Extract links
    const links = await extractLinks(page);
    
    return {
      url,
      title: metadata.title,
      description: metadata.description,
      content,
      links
    };
  } catch (error) {
    debug(`Error extracting content from ${url}:`, error);
    throw error;
  } finally {
    await page.close();
  }
}

/**
 * Perform a deep search by following links to a specified depth
 */
export async function performDeepSearch(initialUrls: string[], options: DeepSearchOptions = {}): Promise<PageContent[]> {
  const depth = options.depth || 1;
  const maxPages = options.maxPages || 5;
  const visitedUrls = new Set<string>();
  const results: PageContent[] = [];
  
  debug(`Starting deep search with depth ${depth} and max pages ${maxPages}`);
  
  // Queue of URLs to visit, with their current depth
  const queue: Array<{ url: string, currentDepth: number }> = initialUrls.map(url => ({ 
    url, 
    currentDepth: 1 
  }));
  
  while (queue.length > 0 && results.length < maxPages) {
    const { url, currentDepth } = queue.shift()!;
    
    // Skip if already visited or exceeds max pages
    if (visitedUrls.has(url) || results.length >= maxPages) {
      continue;
    }
    
    visitedUrls.add(url);
    
    try {
      // Extract content from the current URL
      const pageContent = await extractContentFromUrl(url);
      results.push(pageContent);
      
      // If we haven't reached the maximum depth, add linked pages to the queue
      if (currentDepth < depth) {
        const linkedPages = pageContent.links
          .filter(link => !visitedUrls.has(link.url))
          .map(link => ({
            url: link.url,
            currentDepth: currentDepth + 1
          }));
        
        // Add linked pages to the queue
        queue.push(...linkedPages);
      }
    } catch (error) {
      debug(`Error processing ${url}:`, error);
      // Continue with the next URL
    }
  }
  
  return results;
}