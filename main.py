import os
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
import re
import concurrent.futures
from urllib.parse import urljoin, urlparse
from typing import List, Dict
from langchain_groq import ChatGroq
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from flask import Flask, request, jsonify
from flask_cors import CORS
from tqdm import tqdm  # Import tqdm (if not already imported)
import json  #Import JSON
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize ChromaDB (moved outside the route handlers)
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    name="web_content",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# Initialize Groq LLM (moved outside the route handlers)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=4096,
    groq_api_key=GROQ_API_KEY
)

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

    with tqdm(total=max_pages, desc="Crawling URLs") as pbar:
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

                soup = BeautifulSoup(response.text, 'html.parser')

                for link in soup.find_all('a', href=True):
                    href = urljoin(current_url, link['href'])
                    parsed_href = urlparse(href)

                    if parsed_href.netloc == urlparse(homepage).netloc:
                        if href not in visited:
                            to_visit.append(href)
                            all_urls.add(href)
                            pbar.update(1)
                            if len(all_urls) >= max_pages:
                                break

            except Exception as e:
                print(f"Error crawling {current_url}: {e}")
                return {"error": f"Error crawling {current_url}: {e}"}

    return list(all_urls)

def scrape_urls_from_file(urls: List[str]) -> List[Dict]:

    def scrape_single_url(url: str) -> Dict:
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

            return {"url": url, "content": text, "status": "success"}
        except Exception as e:
            return {"url": url, "content": "", "status": f"error: {str(e)}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(
            executor.map(scrape_single_url, urls),
            total=len(urls),
            desc="Scraping URLs"
        ))

    return results

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0

    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_and_store(scraped_data: List[Dict]):
    chroma_docs, chroma_meta, chroma_ids = [], [], []
    doc_counter = 0

    for item in scraped_data:
        if item["status"] == "success" and item["content"]:
            chunks = chunk_text(item["content"])
            for chunk in chunks:
                chroma_docs.append(chunk)
                chroma_meta.append({"url": item["url"]})
                chroma_ids.append(f"doc_{doc_counter}")
                doc_counter += 1

    if chroma_docs:
        chroma_collection.add(
            documents=chroma_docs,
            metadatas=chroma_meta,
            ids=chroma_ids
        )

def query_and_respond(query: str) -> Dict:
    try:
        results = chroma_collection.query(
            query_texts=[query],
            n_results=1
        )

        contexts = results.get('documents', [[]])[0]

        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
        Your answers should be accurate, informative, and directly related to the context provided."""

        user_prompt = f"""Context information is below.
        ---------------------
        {' '.join(contexts)}
        ---------------------
        Given the context information, please answer this question: {query}

        If the context doesn't contain relevant information, please say so instead of making up an answer."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = llm.invoke(messages).content
        return {
            "query": query,
            "response": response,
            "contexts": contexts
        }
    except Exception as e:
        return {
            "query": query,
            "response": f"Error processing query: {e}",
            "contexts": []
        }

# Flask API Endpoints
@app.route('/crawl', methods=['POST'])
def crawl_route():
    try:
        data = request.get_json()
        homepage = data.get('homepage')
        max_pages = data.get('max_pages', 100)  # Default to 100 if not provided

        if not homepage:
            return jsonify({"error": "Homepage URL is required"}), 400

        result = crawl_urls(homepage, max_pages)
        return jsonify(result)

    except Exception as e:
        print(f"Error in /crawl route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/scrape', methods=['POST'])
def scrape_route():
    try:
        data = request.get_json()
        urls = data.get('urls')

        if not urls or not isinstance(urls, list):
            return jsonify({"error": "A list of URLs is required"}), 400

        result = scrape_urls_from_file(urls)
        return jsonify(result)

    except Exception as e:
        print(f"Error in /scrape route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/store', methods=['POST'])
def store_route():
    try:
        data = request.get_json()
        scraped_data = data.get('data')

        if not scraped_data or not isinstance(scraped_data, list):
            return jsonify({"error": "Scraped data (a list of dictionaries) is required"}), 400

        process_and_store(scraped_data)
        return jsonify({"response": "Data stored successfully"})

    except Exception as e:
        print(f"Error in /store route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_route():
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        result = query_and_respond(query)
        return jsonify(result)

    except Exception as e:
        print(f"Error in /query route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8000)))