import requests
import re
import json
import time
from urllib.parse import quote_plus

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

def search_ddg(query):
    """Search DuckDuckGo Lite"""
    try:
        resp = session.get(f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}", timeout=15)
        if resp.status_code != 200:
            return []
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http') and ('kaggle.com' in href or 'github.com' in href or 'medium.com' in href):
                links.append(href)
        return links
    except Exception as e:
        print(f"  Search error: {e}")
        return []

def search_google(query):
    """Search Google"""
    try:
        resp = session.get(f"https://www.google.com/search?q={quote_plus(query)}&num=20", timeout=15)
        urls = re.findall(r'/url\?q=(https?://[^&"]+)', resp.text)
        return [u for u in urls if 'google.com' not in u]
    except Exception as e:
        print(f"  Search error: {e}")
        return []

def fetch_page(url):
    """Fetch a page and return text"""
    try:
        resp = session.get(url, timeout=15)
        return resp.text
    except Exception as e:
        return f"Error: {e}"

# Search queries
queries = [
    "kaggle notebook crypto stock prediction accuracy 70% ensemble xgboost",
    "kaggle bitcoin price direction prediction high accuracy machine learning",
    "kaggle stock market prediction SMOTE oversampling class imbalance accuracy",
    "kaggle crypto forecasting winning solution lightgbm xgboost",
    "kaggle notebook LSTM transformer stock prediction feature engineering",
]

all_urls = {}
for q in queries:
    print(f"\nSearching: {q[:60]}...")
    time.sleep(3)
    urls = search_ddg(q)
    for u in urls:
        if u not in all_urls:
            all_urls[u] = q
            print(f"  Found: {u[:100]}")
    
    if len(urls) < 3:
        # Try Google as backup
        time.sleep(2)
        g_urls = search_google(q)
        for u in g_urls:
            if u not in all_urls and ('kaggle.com' in u or 'github.com' in u):
                all_urls[u] = q
                print(f"  Found (Google): {u[:100]}")

print(f"\n\n{'='*80}")
print(f"Total unique URLs: {len(all_urls)}")
for u, q in sorted(all_urls.items()):
    print(f"\n[{q[:50]}...]")
    print(f"  {u[:120]}")
