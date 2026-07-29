import asyncio
from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from sentiment_analysis.api import router as api_router, web_router
from sentiment_analysis.core import settings, setup_logging, register_exception_handlers
from sentiment_analysis.db import Base, engine, SessionLocal, ScraperConfig
from sentiment_analysis.scraper import scheduler, sync_jobs

# On Windows, we configure the Proactor event loop to ensure proper asynchronous operations.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def seed_database():
    """Populates the database with default scraper configurations if empty."""
    db = SessionLocal()
    try:
        exists = db.query(ScraperConfig).first()
        if not exists:
            logger.info("Database is empty. Seeding with default scraper configurations...")
            for site in settings.SCRAPER_WEBSITES:
                config = ScraperConfig(
                    name=site["name"],
                    base_url=site["base_url"],
                    scrape_url=site["scrape_url"],
                    schedule_cron=site.get("schedule_cron", "0 9 * * 1"),
                    is_active=site.get("is_active", True)
                )
                db.add(config)
            db.commit()
            logger.info("Database seeding completed successfully.")
    except Exception as e:
        logger.error(f"Failed to seed default configurations: {str(e)}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles microservice startup and shutdown lifecycles."""
    logger.info("Service starting up. Running database checks and synchronization...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema synchronized successfully.")
        seed_database()
    except Exception as e:
        logger.critical(f"Database connection or sync failed on startup: {str(e)}")
        logger.warning("Starting without an active database backend. Persistence features will not work.")

    logger.info("Initializing scraper scheduler...")
    try:
        sync_jobs()
        scheduler.start()
        logger.info("Scraper scheduler started successfully.")
    except Exception as e:
        logger.error(f"Could not start the scraper scheduler: {str(e)}")

    yield
    
    logger.info("Service shutting down. Cleaning up active resources...")
    try:
        scheduler.shutdown()
        logger.info("Scraper scheduler shut down successfully.")
    except Exception as e:
        logger.error(f"Failed to clean up scheduler: {str(e)}")

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    setup_logging()
    logger.info("Initializing Sentiment Analysis Microservice...")

    app = FastAPI(
        title="ENR Sentiment Analysis Service",
        description="Enterprise-grade NLP & Sentiment Analysis service for Energy, Petrochemicals, and Biofuels.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
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

