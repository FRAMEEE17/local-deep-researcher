#!/usr/bin/env python3
import requests
JINA_API_KEY = "jina_c97720348ed44592b1f741e5f2015fe6CqM1e0XNHeD792_n-uoSzC2oMazQ"

def test_jina_scraping():
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    
    # Test 1: ArXiv HTML page (what you're giving)
    print("=== Testing ArXiv HTML Page ===")
    html_url = "https://r.jina.ai/https://arxiv.org/html/2410.21338v2"
    response1 = requests.get(html_url, headers=headers)
    print(f"Length: {len(response1.text)} characters")
    print(f"Content preview: {response1.text[:300]}...")
    print()
    
    # Test 2: ArXiv PDF URL (what should be scraped)
    print("=== Testing ArXiv PDF URL ===")
    pdf_url = "https://r.jina.ai/https://arxiv.org/pdf/2410.21338v2.pdf"
    response2 = requests.get(pdf_url, headers=headers)
    print(f"Length: {len(response2.text)} characters")
    print(f"Content preview: {response2.text[:300]}...")
    print()
    
    # Test 3: ArXiv Abstract page
    print("=== Testing ArXiv Abstract Page ===")
    abs_url = "https://r.jina.ai/https://arxiv.org/abs/2410.21338"
    response3 = requests.get(abs_url, headers=headers)
    print(f"Length: {len(response3.text)} characters")
    print(f"Content preview: {response3.text[:300]}...")

if __name__ == "__main__":
    test_jina_scraping()