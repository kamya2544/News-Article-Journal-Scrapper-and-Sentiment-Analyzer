from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentiment_analysis.core import ScrapingException
from sentiment_analysis.db import ScraperConfig, ScraperLog
from sentiment_analysis.services import ScraperService

def test_create_scraper_config(client):
    payload = {
        "name": "Reuters Test",
        "base_url": "https://www.reuters.com",
        "scrape_url": "https://www.reuters.com/business/energy/lng/",
        "schedule_cron": "0 9 * * 1",
        "is_active": True
    }
    response = client.post("/api/v1/scraper/configs", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Reuters Test"
    assert data["base_url"] == "https://www.reuters.com"
    assert data["scrape_url"] == "https://www.reuters.com/business/energy/lng/"
    assert data["schedule_cron"] == "0 9 * * 1"
    assert data["is_active"] is True
    assert "id" in data

def test_list_scraper_configs(client):
    # Ensure at least one configuration exists
    payload = {
        "name": "List Test",
        "base_url": "https://www.reuters.com",
        "scrape_url": "https://www.reuters.com/business/energy/lng/",
        "schedule_cron": "0 9 * * 1",
        "is_active": True
    }
    client.post("/api/v1/scraper/configs", json=payload)

    response = client.get("/api/v1/scraper/configs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

def test_delete_scraper_config(client):
    # Create config
    payload = {
        "name": "Delete Test",
        "base_url": "https://www.reuters.com",
        "scrape_url": "https://www.reuters.com/business/energy/lng/",
        "schedule_cron": "0 9 * * 1",
        "is_active": True
    }
    create_resp = client.post("/api/v1/scraper/configs", json=payload)
    config_id = create_resp.json()["id"]

    # Delete
    response = client.delete(f"/api/v1/scraper/configs/{config_id}")
    assert response.status_code == 204

    # Verify deleted
    get_resp = client.get("/api/v1/scraper/configs")
    configs = get_resp.json()
    assert not any(c["id"] == config_id for c in configs)

def test_get_scraper_logs(client):
    response = client.get("/api/v1/scraper/logs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@patch("sentiment_analysis.scraper.httpx.AsyncClient")
@patch("sentiment_analysis.services.ScraperService.scrape_url")
@patch("sentiment_analysis.services.NlpPipelineOrchestrator.run_url_pipeline")
def test_manual_run_scraper(mock_run_pipeline, mock_scrape_url, mock_http_client, client):
    # Create a config
    payload = {
        "name": "Manual Run Test",
        "base_url": "https://www.reuters.com",
        "scrape_url": "https://www.reuters.com/business/energy/lng/",
        "schedule_cron": "0 9 * * 1",
        "is_active": True
    }
    create_resp = client.post("/api/v1/scraper/configs", json=payload)
    config_id = create_resp.json()["id"]

    # Mock Listing Page HTTP request
    mock_response = MagicMock() if not hasattr(mock_http_client, "AsyncMock") else AsyncMock()
    mock_response.status_code = 200
    mock_response.text = '<html><body><a href="https://www.reuters.com/business/energy/lng-article-1">Article 1</a></body></html>'
    
    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_response
    mock_http_client.return_value.__aenter__.return_value = mock_client_instance
 
    # Mock scrape_url
    mock_scrape_url.return_value = {
        "title": "Article 1 Title",
        "author": "Author 1",
        "published_date": "2026-07-28",
        "source": "reuters.com",
        "url": "https://www.reuters.com/business/energy/lng-article-1",
        "content": "Chevron boosts biofuel output in modern refinery partnership.",
        "images": []
    }

    # Run
    response = client.post(f"/api/v1/scraper/configs/{config_id}/run")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["articles_scraped"] == 1
    assert data["articles_updated"] == 0


@pytest.mark.asyncio
async def test_scraper_paywall_detection():
    scraper = ScraperService()
    
    # Mocking HTML containing paywall indicators
    paywall_html = """
    <html>
        <body>
            <h1>Ares buy Leonard Green</h1>
            <div class="article-body">
                <p>Subscribe to unlock this article and try unlimited access.</p>
                <p>Only Rs100 for 4 weeks then Rs4499 per month.</p>
            </div>
        </body>
    </html>
    """
    
    with patch("sentiment_analysis.services.httpx.AsyncClient") as mock_http_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = paywall_html
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_http_client.return_value.__aenter__.return_value = mock_client_instance
        
        with pytest.raises(ScrapingException) as exc_info:
            await scraper.scrape_url("https://www.ft.com/content/12345")
        
        assert "Subscription paywall" in str(exc_info.value)

