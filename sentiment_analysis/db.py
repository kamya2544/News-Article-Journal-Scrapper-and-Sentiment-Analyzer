import chromadb
from chromadb.config import Settings as ChromaSettings
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from loguru import logger
from sentiment_analysis.core import settings, get_ist_time

# --- 1. SQL DATABASE CONFIGURATION & SESSIONS ---

# Create engine with MySQL
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 2. CHROMADB VECTOR CLIENT FACTORY ---

def get_chroma_client():
    try:
        if settings.CHROMADB_HOST and settings.CHROMADB_PORT:
            logger.info(f"Connecting to remote ChromaDB at {settings.CHROMADB_HOST}:{settings.CHROMADB_PORT}")
            client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT,
                settings=ChromaSettings(allow_reset=True)
            )
        else:
            logger.info(f"Connecting to persistent local ChromaDB at {settings.CHROMADB_PERSIST_DIRECTORY}")
            client = chromadb.PersistentClient(
                path=settings.CHROMADB_PERSIST_DIRECTORY,
                settings=ChromaSettings(allow_reset=True)
            )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {str(e)}")
        logger.warning("Falling back to ephemeral memory-based ChromaDB client.")
        return chromadb.EphemeralClient()


# --- 3. RELATIONAL MODELS (SQLAlchemy) ---

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=True)
    author = Column(String(255), nullable=True)
    source = Column(String(255), nullable=True)
    published_date = Column(String(100), nullable=True)
    url = Column(String(2048), nullable=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=get_ist_time)

    # OCR specific columns
    images_found = Column(Integer, default=0)
    images_processed = Column(Integer, default=0)
    ocr_text = Column(Text, nullable=True)
    ocr_success = Column(Boolean, default=False)

    # Relationships
    overall_sentiment = relationship("OverallSentiment", back_populates="article", uselist=False, cascade="all, delete-orphan")
    aspect_sentiments = relationship("AspectSentiment", back_populates="article", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="article", cascade="all, delete-orphan")
    topics = relationship("Topic", back_populates="article", cascade="all, delete-orphan")
    drivers = relationship("SentimentDriver", back_populates="article", cascade="all, delete-orphan")


class OverallSentiment(Base):
    __tablename__ = "overall_sentiments"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_ist_time)

    # Relationships
    article = relationship("Article", back_populates="overall_sentiment")


class AspectSentiment(Base):
    __tablename__ = "aspect_sentiments"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    aspect = Column(String(100), nullable=False)
    sentiment = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_ist_time)

    # Relationships
    article = relationship("Article", back_populates="aspect_sentiments")


class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    text = Column(String(255), nullable=False)
    label = Column(String(100), nullable=False)  # ORG, PERSON, GPE, etc.
    created_at = Column(DateTime, default=get_ist_time)

    # Relationships
    article = relationship("Article", back_populates="entities")


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=get_ist_time)

    # Relationships
    article = relationship("Article", back_populates="topics")


class SentimentDriver(Base):
    __tablename__ = "sentiment_drivers"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    driver_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_ist_time)

    # Relationships
    article = relationship("Article", back_populates="drivers")


class ScraperConfig(Base):
    __tablename__ = "scraper_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    base_url = Column(String(1024), nullable=False)
    scrape_url = Column(String(2048), nullable=False)
    schedule_cron = Column(String(255), nullable=False, default="0 9 * * 1")  # e.g., Every Monday at 9 AM IST
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=get_ist_time)
    updated_at = Column(DateTime, default=get_ist_time, onupdate=get_ist_time)

    logs = relationship("ScraperLog", back_populates="config", cascade="all, delete-orphan")


class ScraperLog(Base):
    __tablename__ = "scraper_logs"

    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("scraper_configs.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(50), nullable=False)  # running, success, failed
    articles_scraped = Column(Integer, default=0)
    articles_updated = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=get_ist_time)
    completed_at = Column(DateTime, nullable=True)

    config = relationship("ScraperConfig", back_populates="logs")
