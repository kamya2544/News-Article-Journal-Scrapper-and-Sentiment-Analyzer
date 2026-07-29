from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from typing import List
from loguru import logger
from datetime import datetime, timezone
import os

from sentiment_analysis.core import settings, get_ist_time
from sentiment_analysis.db import get_db, Article, ScraperConfig, ScraperLog
from sentiment_analysis.services import NlpPipelineOrchestrator, ArticleRepository, SentimentRepository
from sentiment_analysis.providers import get_provider_factory
from sentiment_analysis.schemas import (
    SentimentAnalysisResponseSchema,
    ErrorResponse,
    ScraperConfigCreate,
    ScraperConfigResponse,
    ScraperLogResponse
)
from sentiment_analysis.scraper import update_job_schedule, remove_job, ScraperOrchestrator

router = APIRouter(prefix="/api/v1")

# Dependencies Injection Helpers

def get_article_repository(db: Session = Depends(get_db)) -> ArticleRepository:
    return ArticleRepository(db)

def get_sentiment_repository(db: Session = Depends(get_db)) -> SentimentRepository:
    return SentimentRepository(db)

def get_pipeline_orchestrator(
    article_repo: ArticleRepository = Depends(get_article_repository),
    sentiment_repo: SentimentRepository = Depends(get_sentiment_repository)
) -> NlpPipelineOrchestrator:
    return NlpPipelineOrchestrator(
        article_repo=article_repo,
        sentiment_repo=sentiment_repo,
        provider_factory=get_provider_factory()
    )


# API Endpoints

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Microservice health check",
    response_description="Returns success if microservice is healthy"
)
async def health_check():
    logger.debug("Health check invoked.")
    return {
        "status": "healthy",
        "timestamp": get_ist_time().isoformat()
    }


# --- Scraper Configurations & Logs API ---

@router.get("/scraper/configs", response_model=List[ScraperConfigResponse], summary="List all scraper configurations")
def list_scraper_configs(db: Session = Depends(get_db)):
    return db.query(ScraperConfig).all()

@router.post("/scraper/configs", response_model=ScraperConfigResponse, status_code=status.HTTP_201_CREATED, summary="Create a new scraper configuration")
def create_scraper_config(payload: ScraperConfigCreate, db: Session = Depends(get_db)):
    config = ScraperConfig(
        name=payload.name,
        base_url=str(payload.base_url),
        scrape_url=str(payload.scrape_url),
        schedule_cron=payload.schedule_cron,
        is_active=payload.is_active
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    
    # Update scheduler
    update_job_schedule(config)
    
    return config

@router.delete("/scraper/configs/{id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a scraper configuration")
def delete_scraper_config(id: int, db: Session = Depends(get_db)):
    config = db.query(ScraperConfig).filter(ScraperConfig.id == id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Scraper configuration not found")
    
    db.delete(config)
    db.commit()
    
    # Remove from scheduler
    remove_job(id)
    
    return None

@router.post("/scraper/configs/{id}/run", response_model=ScraperLogResponse, summary="Manually trigger a scraper config run immediately")
async def run_scraper_config(
    id: int,
    db: Session = Depends(get_db),
    orchestrator: NlpPipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    config = db.query(ScraperConfig).filter(ScraperConfig.id == id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Scraper configuration not found")
        
    scraper_orchestrator = ScraperOrchestrator(db, orchestrator)
    log_entry = await scraper_orchestrator.run_scraper(id)
    return log_entry

@router.get("/scraper/logs", response_model=List[ScraperLogResponse], summary="Retrieve execution logs for all scraper configs")
def get_scraper_logs(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(ScraperLog).order_by(ScraperLog.started_at.desc()).limit(limit).all()


@router.get("/articles", response_model=List[SentimentAnalysisResponseSchema], summary="Retrieve all analyzed articles")
def get_articles(
    limit: int = 50,
    db: Session = Depends(get_db),
    orchestrator: NlpPipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    import time
    articles = db.query(Article).order_by(Article.created_at.desc()).limit(limit).all()
    return [orchestrator.map_db_to_schema(art, time.time()) for art in articles]


web_router = APIRouter()

@web_router.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/dashboard")

@web_router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
