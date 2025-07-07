import requests
from typing import List, Dict, Any

class NewsCollector:
    """
    Generic template for fetching news from APIs.
    Extend this class for specific news APIs.
    """
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def fetch_news(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch news articles from the API.
        Args:
            params: Dictionary of query parameters for the API request.
        Returns:
            List of news articles (as dictionaries).
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(self.base_url, params=params, headers=headers)
            if response.status_code == 200:
                return response.json().get("articles", [])
            else:
                response.raise_for_status()
                return []
        except Exception as e:
            # Log the error if needed
            return []

# Example usage (to be replaced with actual API details):
# collector = NewsCollector(api_key="YOUR_API_KEY", base_url="https://newsapi.org/v2/everything")
# news = collector.fetch_news({"q": "stock market", "language": "en"})
# print(news)
