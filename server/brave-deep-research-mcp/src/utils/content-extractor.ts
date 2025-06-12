import { Page } from 'puppeteer';
import { debug } from './config.js';

/**
 * Extract the main content from a webpage
 * Attempts to identify and extract the main content, excluding navigation, footers, etc.
 */
export async function extractMainContent(page: Page): Promise<string> {
  debug(`Extracting main content from ${page.url()}`);
  
  return await page.evaluate(() => {
    // Define potential content selectors - ordered by priority
    const contentSelectors = [
      'article',
      'main',
      '.content',
      '#content',
      '.post-content',
      '.entry-content',
      '.article-content',
      '.article-body',
      '.post-body',
    ];
    
    // Define elements to exclude
    const excludeSelectors = [
      'nav',
      'header',
      'footer',
      '.navigation',
      '.menu',
      '.sidebar',
      '.ads',
      '.advertisement',
      '.comments',
      '.related',
      '.share',
      '.social',
    ];
    
    // Try to find main content using selectors
    for (const selector of contentSelectors) {
      const element = document.querySelector(selector);
      if (element && element.textContent && element.textContent.trim().length > 100) {
        return element.textContent.trim();
      }
    }
    
    // If no main content is found, use the body but exclude common non-content elements
    const body = document.body;
    const clonedBody = body.cloneNode(true) as HTMLBodyElement;
    
    // Remove excluded elements
    excludeSelectors.forEach(selector => {
      const elements = clonedBody.querySelectorAll(selector);
      elements.forEach(el => el.parentNode?.removeChild(el));
    });
    
    // Return cleaned content
    return clonedBody.textContent?.trim() || "No content found";
  });
}

/**
 * Extract links from a page
 * Returns an array of objects with url and text properties
 */
export async function extractLinks(page: Page): Promise<Array<{ url: string, text: string }>> {
  debug(`Extracting links from ${page.url()}`);
  
  return await page.evaluate(() => {
    const links: Array<{ url: string, text: string }> = [];
    
    // Get all links in the document
    const anchorElements = document.querySelectorAll('a[href]');
    
    anchorElements.forEach(anchor => {
      const href = anchor.getAttribute('href');
      
      // Skip if no href or it's a special link (javascript:, mailto:, tel:, anchor)
      if (!href || href.startsWith('javascript:') || href.startsWith('mailto:') || 
          href.startsWith('tel:') || href.startsWith('#')) {
        return;
      }
      
      // Convert relative URLs to absolute
      const url = new URL(href, window.location.href).href;
      const text = anchor.textContent?.trim() || url;
      
      // Only include links with text and from the same domain
      if (text && url.startsWith(window.location.origin)) {
        links.push({ url, text });
      }
    });
    
    return links;
  });
}

/**
 * Extract title and description from a page
 */
export async function extractMetadata(page: Page): Promise<{ title: string, description: string }> {
  debug(`Extracting metadata from ${page.url()}`);
  
  return await page.evaluate(() => {
    const title = document.title || "";
    
    // Try to get meta description
    const metaDescription = document.querySelector('meta[name="description"]')?.getAttribute('content') || 
                            document.querySelector('meta[property="og:description"]')?.getAttribute('content') || 
                            "";
    
    return { title, description: metaDescription };
  });
}