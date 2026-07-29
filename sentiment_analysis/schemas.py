from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sentiment_analysis.core import SentimentLabel, get_ist_time

# --- Response Component Schemas ---

class ArticleMetadataSchema(BaseModel):
    title: Optional[str] = Field(None, description="Title of the article")
    author: Optional[str] = Field(None, description="Author of the article")
    source: Optional[str] = Field(None, description="Source domain or site name")
    published_date: Optional[str] = Field(None, description="Date the article was published")
    url: Optional[str] = Field(None, description="Origin URL of the article")


class OverallSentimentSchema(BaseModel):
    label: SentimentLabel = Field(..., description="Overall sentiment classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Inference confidence score")
    reasoning: Optional[str] = Field(None, description="Detailed explanation of the sentiment")


class AspectSentimentSchema(BaseModel):
    aspect: str = Field(..., description="Configured aspect/domain, e.g. Biofuels, Petrochemicals")
    sentiment: SentimentLabel = Field(..., description="Sentiment assigned to the aspect")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Aspect sentiment confidence score")
    reason: Optional[str] = Field(None, description="Reasoning/evidence for this aspect's sentiment classification")


class OCRMetadataSchema(BaseModel):
    images_found: int = Field(0, description="Total number of images detected in the article")
    images_processed: int = Field(0, description="Number of images processed through OCR")
    ocr_text: List[str] = Field(default_factory=list, description="Extracted text from images")
    success: bool = Field(False, description="Flag indicating if OCR finished without errors")


class EntitySchema(BaseModel):
    text: str = Field(..., description="Extracted entity text")
    label: str = Field(..., description="Type of entity (e.g. Org, Country, Product, Chemical)")


class ExecutionMetadataSchema(BaseModel):
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_name: str = Field(..., description="Active model name")
    model_version: str = Field(..., description="Active model version/specification")
    timestamp: datetime = Field(default_factory=get_ist_time, description="IST Processing timestamp")


# --- Main Response Schema ---

class SentimentAnalysisResponseSchema(BaseModel):
    article: ArticleMetadataSchema
    overall_sentiment: OverallSentimentSchema
    summary: Optional[str] = Field(None, description="Concise executive business summary")
    topics: List[str] = Field(default_factory=list, description="Extracted topics")
    entities: List[EntitySchema] = Field(default_factory=list, description="Extracted key business entities")
    aspect_sentiment: List[AspectSentimentSchema] = Field(default_factory=list, description="Aspect-wise sentiment results")
    sentiment_drivers: List[str] = Field(default_factory=list, description="Extracted sentiment driver reasons")
    ocr: OCRMetadataSchema
    metadata: ExecutionMetadataSchema


# --- Error Schemas ---

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail


# --- Scraper Schemas ---

class ScraperConfigCreate(BaseModel):
    name: str = Field(..., description="Website name / source name")
    base_url: str = Field(..., description="Base URL of the website")
    scrape_url: str = Field(..., description="Target listing/feed URL to scrape")
    schedule_cron: str = Field("0 9 * * 1", description="Cron expression for scheduling (default: Monday 9 AM IST)")
    is_active: bool = Field(True, description="Whether schedule is active")

    @field_validator("base_url", "scrape_url")
    @classmethod
    def validate_urls(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class ScraperLogResponse(BaseModel):
    id: int
    config_id: int
    status: str
    articles_scraped: int
    articles_updated: int
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ScraperConfigResponse(BaseModel):
    id: int
    name: str
    base_url: str
    scrape_url: str
    schedule_cron: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
