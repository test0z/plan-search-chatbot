import os
import requests
import asyncio
import httpx
import redis
import scrapy
from urllib.parse import urljoin
from typing import List, Tuple, Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SearchCrawler(ABC):
    """
    Abstract base class for handling web search and content extraction from search results.
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        """
        Initialize the SearchCrawler with the given configuration.
        
        Args:
            redis_config: Configuration dictionary for Redis connection
        """
        self.use_redis = os.getenv("REDIS_USE", "False").lower() == "true"
        
        if not self.use_redis:
            logger.info("Redis usage is disabled by configuration")
            self.redis_client = None
            return
            
        default_redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "password": os.getenv("REDIS_PASSWORD", ""),
            "db": int(os.getenv("REDIS_DB", 0)),
            "decode_responses": True
        }
        
        redis_config = redis_config or default_redis_config
        
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(**redis_config)
            self.redis_client.ping()
            logger.info("Redis cache connection established successfully")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Redis initialization error: {str(e)}. Caching disabled.")
            self.redis_client = None
            
        self.cache_expiration = int(os.getenv("REDIS_CACHE_EXPIRED_SECOND", 604800))
    
    @abstractmethod
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a web search using a specific search engine.
        
        Args:
            query: The search query
            
        Returns:
            A list of search result items
        """
        pass
    
    async def extract_contexts_async(self, url_snippet_tuples: List[Tuple[str, str]]) -> List[str]:
        """
        Asynchronously extract content from a list of URLs with their snippets.
        
        Args:
            url_snippet_tuples: List of (url, snippet) pairs to process
            
        Returns:
            List of extracted contents
        """
        async def fetch_and_cache(url: str, snippet: str) -> str:
            if self.redis_client:
                try:
                    cached_text = self.redis_client.get(url)
                    if cached_text:
                        logger.info(f"Retrieved content from Redis cache for URL: {url}")
                        return cached_text
                except Exception as e:
                    logger.warning(f"Redis get operation failed: {str(e)}")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            try:
                async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                    try:
                        response = await client.get(url, headers=headers)
                        response.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 302 and "location" in e.response.headers:
                            redirect_url = e.response.headers["location"]
                            if not redirect_url.startswith("http"):
                                redirect_url = urljoin(url, redirect_url)
                            try:
                                response = await client.get(redirect_url, headers=headers)
                                response.raise_for_status()
                            except Exception as e2:
                                logger.error(f"Redirect request failed: {e2}")
                                return f"{snippet} "
                        else:
                            logger.error(f"Request failed: {e}")
                            return f"{snippet} "
                    except httpx.HTTPError as e:
                        logger.error(f"Request failed: {e}")
                        return f"{snippet} "
                    
                    selector = scrapy.Selector(text=response.text)
                    
                    paragraphs = [p.strip() for p in selector.css('p::text, p *::text').getall() if p.strip()]
                    
                    filtered_paragraphs = []
                    seen_content = set()
                    for p in paragraphs:
                        if len(p) < 5:
                            continue
                        if p in seen_content:
                            continue
                        seen_content.add(p)
                        filtered_paragraphs.append(p)
                    
                    text = "\n".join(filtered_paragraphs)
                    
                    if not text:
                        content_texts = [t.strip() for t in selector.css(
                            'article::text, article *::text, .content::text, .content *::text, '
                            'main::text, main *::text'
                        ).getall() if t.strip()]
                        
                        if content_texts:
                            text = "\n".join(content_texts)
                    
                    snippet_text = f"{snippet}: {text}"
                    
                    if self.redis_client and len(text) > 200:
                        try:
                            self.redis_client.set(url, snippet_text, ex=self.cache_expiration)
                            logger.info(f"Stored content in Redis cache for URL: {url}")
                        except Exception as e:
                            logger.warning(f"Redis cache operation failed: {str(e)}")
                    
                    return snippet_text
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                return f"{snippet} [Error: {str(e)}]"
        
        tasks = [asyncio.create_task(fetch_and_cache(url, snippet)) 
                for url, snippet in url_snippet_tuples]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing URL {url_snippet_tuples[i][0]}: {str(result)}")
                processed_results.append(f"{url_snippet_tuples[i][1]} [Processing Error]")
            else:
                processed_results.append(result)
                
        return processed_results


class GoogleSearchCrawler(SearchCrawler):
    """
    Google-specific implementation of the SearchCrawler.
    Uses Google Custom Search API for performing searches.
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        """
        Initialize the GoogleSearchCrawler with the given configuration.
        
        Args:
            redis_config: Configuration dictionary for Redis connection
        """
        super().__init__(redis_config)
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.google_max_result = os.getenv("GOOGLE_MAX_RESULT")
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a Google search using Custom Search API.
        
        Args:
            query: The search query
            
        Returns:
            A list of search result items
        """
        if not self.google_api_key or not self.google_cse_id:
            logger.error("Google API credentials are missing")
            return []
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query + " -filetype:pdf",
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "num": self.google_max_result,
            "locale": "ko-KR",
            "filter": "1",
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            return results.get("items", [])
        except requests.RequestException as e:
            logger.error(f"Google search API error: {str(e)}")
            return []

class BingSearchCrawler(SearchCrawler):
    """
    Bing-specific implementation of the SearchCrawler.
    Uses Bing Search API for performing searches.
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        """
        Initialize the BingSearchCrawler with the given configuration.
        
        Args:
            redis_config: Configuration dictionary for Redis connection
        """
        super().__init__(redis_config)
        self.bing_endpoint = os.getenv("BING_ENDPOINT")
        self.bing_api_key = os.getenv("BING_API_KEY")
        self.bing_custom_config_id = os.getenv("BING_CUSTOM_CONFIG_ID")
        self.bing_max_result = int(os.getenv("BING_MAX_RESULT", "10"))
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a Bing search using Bing Search API.
        
        Args:
            query: The search query
            
        Returns:
            A list of search result items
        """
        if not self.bing_api_key:
            logger.error("Bing API key is missing")
            return []
            
        
        if self.bing_custom_config_id:
            url = "https://api.bing.microsoft.com/v7.0/custom/search"
        else:
            url = "https://api.bing.microsoft.com/v7.0/search"

        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key
        }
        params = {
            "q": query + " -filetype:pdf",
            "count": self.bing_max_result,
            "mkt": "ko-KR",
            "responseFilter": "Webpages",
        }
        if self.bing_custom_config_id:
            params.update({
                "customconfig": self.bing_custom_config_id,
            })
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            results = response.json()
            
            # Convert Bing format to match Google format for consistency
            formatted_results = []
            if "webPages" in results and "value" in results["webPages"]:
                for item in results["webPages"]["value"]:
                    formatted_results.append({
                        "link": item.get("url"),
                        "snippet": item.get("name"),
                        # "snippet": item.get("snippet"),
                        # "displayLink": item.get("displayUrl")
                    })
            
            return formatted_results
        except requests.RequestException as e:
            logger.error(f"Bing search API error: {str(e)}")
            return []
        

        
