import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from contextlib import asynccontextmanager

from sentiment_analysis.core import settings, setup_logging, register_exception_handlers
from sentiment_analysis.db import Base, engine, SessionLocal, ScraperConfig
from sentiment_analysis.api import router as api_router, web_router
from sentiment_analysis.scraper import scheduler, sync_jobs

def seed_database():
    db = SessionLocal()
    try:
        exists = db.query(ScraperConfig).first()
        if not exists:
            logger.info("Seeding database with default scraper configurations...")
            reuters_config = ScraperConfig(
                name="Reuters Energy",
                base_url="https://www.reuters.com",
                scrape_url="https://www.reuters.com/business/energy/lng/",
                schedule_cron="0 9 * * 1",
                is_active=True
            )
            ft_config = ScraperConfig(
                name="Financial Times Energy",
                base_url="https://www.ft.com",
                scrape_url="https://www.ft.com/energy",
                schedule_cron="0 9 * * 1",
                is_active=True
            )
            db.add(reuters_config)
            db.add(ft_config)
            db.commit()
            logger.info("Seeding completed successfully.")
    except Exception as e:
        logger.error(f"Failed to seed default configurations: {str(e)}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Service starting up. Executing database checks...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("MySQL/SQLite database schema synchronized successfully.")
        seed_database()
    except Exception as e:
        logger.critical(f"Database connection/sync failed on startup: {str(e)}")
        logger.warning("Application starting without initialized SQL backend. Persistence functions will fail.")

    logger.info("Starting scraper scheduler...")
    try:
        sync_jobs()
        scheduler.start()
        logger.info("Scraper scheduler started successfully.")
    except Exception as e:
        logger.error(f"Failed to start scraper scheduler: {str(e)}")

    yield
    logger.info("Service shutting down. Closing connections...")
    try:
        scheduler.shutdown()
        logger.info("Scraper scheduler shut down successfully.")
    except Exception as e:
        logger.error(f"Failed to shut down scheduler: {str(e)}")

def create_app() -> FastAPI:
    setup_logging()
    logger.info("Initializing Sentiment Analysis Microservice...")

    app = FastAPI(
        title="ENR Sentiment Analysis Service",
        description="Enterprise-grade NLP & Sentiment Analysis service for Energy, Petrochemicals, and Biofuels.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(api_router)
    app.include_router(web_router)

    return app

app = create_app()
