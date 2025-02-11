from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin, urlparse
from typing import List
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

app = FastAPI()

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def crawl_urls(homepage: str, max_pages: int = 100) -> List[str]:
    visited = set()
    to_visit = [homepage]
    all_urls = set()

    while to_visit and len(all_urls) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        visited.add(current_url)
        try:
            response = requests_retry_session().get(current_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            print(f"Crawling: {current_url}")

            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = urljoin(current_url, link['href'])
                parsed_href = urlparse(href)

                if parsed_href.netloc == urlparse(homepage).netloc:
                    if href not in visited:
                        to_visit.append(href)
                        all_urls.add(href)
                        if len(all_urls) >= max_pages:
                            break
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")

    return list(all_urls)

def scrape_page(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests_retry_session().get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,?!-]', '', text)
        return text
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/scrape")
async def scrape_endpoint(url: str, command: str, query: str):
    text = scrape_page(url)  # Call scrape_page *without* try-except here
    if "Error:" in text: # Check the return value for an error string.
        raise HTTPException(status_code=500, detail=text)  # Raise the exception
    return {"text": text, "command": command, "query": query}