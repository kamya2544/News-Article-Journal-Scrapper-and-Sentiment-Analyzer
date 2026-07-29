import json
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Any, List

import pytz
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

IST = pytz.timezone("Asia/Kolkata")

def get_ist_time() -> datetime:
    return datetime.now(IST)

# --- 1. CONFIGURATION & SETTINGS ---

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # AI Models
    MODEL_NAME: str = "ProsusAI/finbert"
    NER_MODEL: str = "en_core_web_sm"
    OCR_ENGINE: str = "easyocr"

    # Databases
    DATABASE_URL: str = "mysql+pymysql://root:password@localhost:3306/sentiment_db"
    CHROMADB_PERSIST_DIRECTORY: str = "./chromadb_data"
    CHROMADB_HOST: str = ""
    CHROMADB_PORT: str = ""

    # Feature Flags
    ENABLE_OCR: bool = True
    ENABLE_SUMMARY: bool = True
    ENABLE_NER: bool = True
    ENABLE_ASPECT_SENTIMENT: bool = True
    MAX_ARTICLES_PER_FEED: int = 0 # 0 or negative for unlimited

    # Allowed Aspects
    ALLOWED_ASPECTS: Any = [
        "Biofuels",
        "Petrochemicals",
        "Hydrogen",
        "SAF",
        "Renewable Energy",
        "Carbon Capture"
    ]

    # Default Scraper Websites
    SCRAPER_WEBSITES: Any = [
        {
            "name": "Reuters Energy",
            "base_url": "https://www.reuters.com",
            "scrape_url": "https://www.reuters.com/business/energy/lng/",
            "schedule_cron": "0 9 * * 1",
            "is_active": True
        },
        {
            "name": "Financial Times Energy",
            "base_url": "https://www.ft.com",
            "scrape_url": "https://www.ft.com/energy",
            "schedule_cron": "0 9 * * 1",
            "is_active": True
        },
        {
            "name": "Oil & Gas Journal",
            "base_url": "https://www.ogj.com",
            "scrape_url": "https://www.ogj.com",
            "schedule_cron": "0 9 * * 1",
            "is_active": True
        }
    ]

    @field_validator("ALLOWED_ASPECTS", mode="before")
    @classmethod
    def parse_allowed_aspects(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("SCRAPER_WEBSITES", mode="before")
    @classmethod
    def parse_scraper_websites(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception as e:
                logger.error(f"Failed to parse SCRAPER_WEBSITES JSON: {e}")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

def update_env_file_scraper_websites(new_site: dict):
    """Appends or updates a scraper configuration in the .env file."""
    # Avoid modifying the actual .env file during unit testing.
    if "pytest" in sys.modules or "pytest" in sys.argv[0]:
        logger.info("Test environment detected. Skipping .env file write.")
        new_websites_list = list(settings.SCRAPER_WEBSITES)
        for i, site in enumerate(new_websites_list):
            if site.get("name") == new_site["name"] or site.get("scrape_url") == new_site["scrape_url"]:
                new_websites_list[i] = new_site
                settings.SCRAPER_WEBSITES = new_websites_list
                return
        new_websites_list.append(new_site)
        settings.SCRAPER_WEBSITES = new_websites_list
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", ".env")
    
    if not os.path.exists(env_path):
        env_path = os.path.abspath(".env")
        
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    found = False
    new_websites_list = list(settings.SCRAPER_WEBSITES)
    
    exists = False
    for i, site in enumerate(new_websites_list):
        if site.get("name") == new_site["name"] or site.get("scrape_url") == new_site["scrape_url"]:
            new_websites_list[i] = new_site
            exists = True
            break
    if not exists:
        new_websites_list.append(new_site)
        
    settings.SCRAPER_WEBSITES = new_websites_list
    websites_json = json.dumps(new_websites_list)
    new_line = f"SCRAPER_WEBSITES={websites_json}\n"
    
    new_lines = []
    for line in lines:
        if line.strip().startswith("SCRAPER_WEBSITES="):
            new_lines.append(new_line)
            found = True
        else:
            new_lines.append(line)
            
    if not found:
        if not new_lines or not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append("# List of default scraper configurations (JSON format)\n")
        new_lines.append(new_line)
        
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    logger.info(f"Successfully added/updated website '{new_site['name']}' in .env file.")


def remove_env_file_scraper_website(name_or_url: str):
    """Removes a scraper configuration from the .env file."""
    # Avoid modifying the actual .env file during unit testing.
    if "pytest" in sys.modules or "pytest" in sys.argv[0]:
        logger.info("Test environment detected. Skipping .env file write.")
        new_websites_list = [
            site for site in settings.SCRAPER_WEBSITES 
            if site.get("name") != name_or_url and site.get("scrape_url") != name_or_url
        ]
        settings.SCRAPER_WEBSITES = new_websites_list
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", ".env")
    
    if not os.path.exists(env_path):
        env_path = os.path.abspath(".env")
        
    if not os.path.exists(env_path):
        return
        
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_websites_list = [
        site for site in settings.SCRAPER_WEBSITES 
        if site.get("name") != name_or_url and site.get("scrape_url") != name_or_url
    ]
    
    settings.SCRAPER_WEBSITES = new_websites_list
    websites_json = json.dumps(new_websites_list)
    new_line = f"SCRAPER_WEBSITES={websites_json}\n"
    
    new_lines = []
    found = False
    for line in lines:
        if line.strip().startswith("SCRAPER_WEBSITES="):
            new_lines.append(new_line)
            found = True
        else:
            new_lines.append(line)
            
    if found:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info(f"Successfully removed website '{name_or_url}' from .env file.")

# --- 2. ENUMS ---

class InputType(str, Enum):
    TEXT = "text"
    URL = "url"

class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

# --- 3. LOGGING ---

def setup_logging():
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=log_format,
        enqueue=True
    )
    logger.info("Logging initialized successfully.")

# --- 4. EXCEPTIONS & HANDLERS ---

class SentimentAnalysisException(Exception):
    """Base exception for all sentiment analysis errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ScrapingException(SentimentAnalysisException):
    """Raised when scraping fails."""
    def __init__(self, message: str):
        super().__init__(message, status_code=404)

class OCRProcessingException(SentimentAnalysisException):
    """Raised when OCR processing fails."""
    def __init__(self, message: str):
        super().__init__(message, status_code=422)

class DatabaseException(SentimentAnalysisException):
    """Raised when database interactions fail."""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

class ModelException(SentimentAnalysisException):
    """Raised when AI model inference fails."""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(SentimentAnalysisException)
    async def base_exception_handler(request: Request, exc: SentimentAnalysisException):
        logger.error(f"Error executing {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.__class__.__name__,
                    "message": exc.message
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "ValidationError",
                    "message": "Invalid request payload",
                    "details": exc.errors()
                }
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.opt(exception=exc).error(f"Unhandled exception occurred on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "InternalServerError",
                    "message": "An unexpected error occurred on the server."
                }
            }
        )
