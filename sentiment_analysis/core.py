import sys
from typing import List, Any
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
import pytz
from datetime import datetime

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

    @field_validator("ALLOWED_ASPECTS", mode="before")
    @classmethod
    def parse_allowed_aspects(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

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
