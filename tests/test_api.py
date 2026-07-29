import pytest
from sentiment_analysis.db import Article, OverallSentiment, AspectSentiment, Entity, Topic, SentimentDriver

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_articles(client, db_session):
    # Retrieve articles when empty
    response = client.get("/api/v1/articles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

    # Insert an article directly via db session
    article = Article(
        title="Direct DB Test Article",
        author="Test Author",
        source="test.com",
        published_date="2026-07-29",
        url="https://test.com/article",
        content="Chevron refinery output in California increases biofuel development. High demand boosts profit margin.",
        images_found=0,
        images_processed=0,
        ocr_text="",
        ocr_success=True
    )
    db_session.add(article)
    db_session.commit()
    db_session.refresh(article)

    # Add overall sentiment and aspect sentiment
    overall = OverallSentiment(
        article_id=article.id,
        label="Positive",
        confidence=0.95,
        reasoning="Test overall reasoning"
    )
    db_session.add(overall)
    
    aspect = AspectSentiment(
        article_id=article.id,
        aspect="Biofuels",
        sentiment="Positive",
        confidence=0.90,
        reason="Biofuels increased"
    )
    db_session.add(aspect)
    
    entity = Entity(
        article_id=article.id,
        text="Chevron",
        label="Organization"
    )
    db_session.add(entity)
    
    topic = Topic(
        article_id=article.id,
        name="Biofuels"
    )
    db_session.add(topic)
    
    driver = SentimentDriver(
        article_id=article.id,
        driver_text="High demand boosts profit margin."
    )
    db_session.add(driver)
    
    db_session.commit()

    # Retrieve articles and verify structure
    response = client.get("/api/v1/articles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "article" in data[0]
    assert "overall_sentiment" in data[0]
    assert "summary" in data[0]
    assert "topics" in data[0]
    assert "entities" in data[0]
    assert "aspect_sentiment" in data[0]
    assert "sentiment_drivers" in data[0]
    assert "ocr" in data[0]
    assert "metadata" in data[0]
    
    # Assert values
    assert data[0]["overall_sentiment"]["label"] == "Positive"
    assert data[0]["article"]["title"] == "Direct DB Test Article"
    assert len(data[0]["aspect_sentiment"]) > 0
    assert data[0]["aspect_sentiment"][0]["aspect"] == "Biofuels"
    assert data[0]["sentiment_drivers"][0] == "High demand boosts profit margin."
