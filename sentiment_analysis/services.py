from datetime import datetime, timezone
import io
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

from bs4 import BeautifulSoup
import httpx
from loguru import logger
from PIL import Image
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from sqlalchemy.orm import Session
import trafilatura

from sentiment_analysis.core import SentimentLabel, ScrapingException, get_ist_time, settings
from sentiment_analysis.db import (
    Article,
    AspectSentiment,
    Entity,
    OverallSentiment,
    SentimentDriver,
    Topic,
    get_chroma_client
)
from sentiment_analysis.providers import ProviderFactory
from sentiment_analysis.schemas import (
    ArticleMetadataSchema,
    AspectSentimentSchema,
    EntitySchema,
    ExecutionMetadataSchema,
    OCRMetadataSchema,
    OverallSentimentSchema,
    SentimentAnalysisResponseSchema
)

# --- 1. DATA REPOSITORIES ---

class ArticleRepository:
    def __init__(self, db: Session):
        self.db = db
        self.chroma_client = get_chroma_client()
        self.collection_name = "articles_vectors"
        self._init_chromadb_collection()

    def _init_chromadb_collection(self):
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Article contents and OCR embeddings"}
            )
        except Exception as e:
            logger.error(f"Failed to get/create ChromaDB collection: {str(e)}")
            self.collection = None

    def create(self, article: Article) -> Article:
        try:
            self.db.add(article)
            self.db.commit()
            self.db.refresh(article)
            logger.info(f"Article '{article.title}' saved to MySQL with ID: {article.id}")

            # Store in ChromaDB vector database
            if self.collection is not None:
                # Merge OCR text if present
                combined_content = article.content
                if article.ocr_text:
                    combined_content += f"\n[OCR Text]\n{article.ocr_text}"

                metadata = {
                    "mysql_id": article.id,
                    "title": article.title or "",
                    "url": article.url or "",
                    "author": article.author or "",
                    "published_date": article.published_date or ""
                }
                
                self.collection.upsert(
                    ids=[str(article.id)],
                    documents=[combined_content],
                    metadatas=[metadata]
                )
                logger.info(f"Article vector embedded in ChromaDB for ID: {article.id}")
            
            return article
        except Exception as e:
            self.db.rollback()
            logger.error(f"Database error during article creation: {str(e)}")
            raise e

    def get_by_id(self, article_id: int) -> Optional[Article]:
        return self.db.query(Article).filter(Article.id == article_id).first()

    def get_by_url(self, url: str) -> Optional[Article]:
        return self.db.query(Article).filter(Article.url == url).first()


class SentimentRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_analysis(
        self,
        article_id: int,
        overall: Dict[str, Any],
        aspects: List[Dict[str, Any]],
        entities: List[Dict[str, str]],
        topics: List[str],
        drivers: List[str]
    ) -> None:
        try:
            # 1. Save overall sentiment
            db_overall = OverallSentiment(
                article_id=article_id,
                label=overall["label"],
                confidence=overall["confidence"],
                reasoning=overall["reasoning"]
            )
            self.db.add(db_overall)

            # 2. Save aspect sentiments
            for aspect_data in aspects:
                db_aspect = AspectSentiment(
                    article_id=article_id,
                    aspect=aspect_data["aspect"],
                    sentiment=aspect_data["sentiment"],
                    confidence=aspect_data["confidence"],
                    reason=aspect_data["reason"]
                )
                self.db.add(db_aspect)

            # 3. Save entities
            for ent in entities:
                db_ent = Entity(
                    article_id=article_id,
                    text=ent["text"],
                    label=ent["label"]
                )
                self.db.add(db_ent)

            # 4. Save topics
            for t_name in topics:
                db_topic = Topic(
                    article_id=article_id,
                    name=t_name
                )
                self.db.add(db_topic)

            # 5. Save drivers
            for driver_t in drivers:
                db_driver = SentimentDriver(
                    article_id=article_id,
                    driver_text=driver_t
                )
                self.db.add(db_driver)

            self.db.commit()
            logger.info(f"Sentiment analysis records saved successfully for Article ID {article_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save sentiment analysis: {str(e)}")
            raise e


# --- 2. SCRAPING SERVICE ---

class ScraperService:
    def __init__(self):
        # Spoof Googlebot search crawler headers to bypass soft paywalls
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Referer": "https://www.google.com/"
        }

    def _clean_content(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        filtered_lines = []
        noise_keywords = [
            "subscribe to unlock", "unlimited access", "cancel anytime", "terms & conditions",
            "cookie policy", "all rights reserved", "standard digital", "rs100 for 4 weeks",
            "rs4499 per month", "sign in to read", "register to read", "try unlimited access",
            "explore more offers", "complete digital access", "privacy policy", "cookie preference",
            "become a member", "choose your subscription", "subscription plans", "unlock this article",
            "only rs", "per month", "unlock this", "sign in", "register now"
        ]
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            line_lower = line_stripped.lower()
            # Skip if the line contains promotional paywall boilerplate
            if any(keyword in line_lower for keyword in noise_keywords):
                continue
            # Drop typical copyright/Terms headers that are very short
            if len(line_stripped) < 25 and ("copyright" in line_lower or "terms" in line_lower or "privacy" in line_lower or "contact" in line_lower):
                continue
            filtered_lines.append(line_stripped)
        return "\n".join(filtered_lines)

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        logger.info(f"Starting scrap for URL: {url}")
        html = None
        use_playwright = False
        fallback_reason = ""

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=self.headers) as client:
            try:
                response = await client.get(url)
                if response.status_code in (401, 403):
                    fallback_reason = f"HTTP status code {response.status_code}"
                    use_playwright = True
                else:
                    html = response.text
                    lower_content = html.lower()
                    if "datadome" in lower_content:
                        fallback_reason = "DataDome bot protection detected"
                        use_playwright = True
                    elif "captcha" in lower_content:
                        fallback_reason = "Captcha challenge detected"
                        use_playwright = True
                    elif "cloudflare" in lower_content and ("please enable js" in lower_content or "checking your browser" in lower_content):
                        fallback_reason = "Cloudflare anti-bot page detected"
                        use_playwright = True
                    elif response.status_code != 200:
                        raise Exception(f"Failed to fetch webpage, status code: {response.status_code}")
            except Exception as e:
                logger.warning(f"HTTP request error: {str(e)}. Falling back to Playwright.")
                fallback_reason = f"HTTP request exception: {str(e)}"
                use_playwright = True

        if use_playwright:
            logger.info(f"Falling back to Playwright for URL: {url} due to: {fallback_reason}")
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                context = await browser.new_context(
                    user_agent=self.headers["User-Agent"],
                    viewport={"width": 1280, "height": 800}
                )
                page = await context.new_page()
                await Stealth().apply_stealth_async(page)
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    html = await page.content()
                except Exception as pw_err:
                    logger.error(f"Playwright navigation failed: {str(pw_err)}")
                    try:
                        html = await page.content()
                    except Exception:
                        raise Exception(f"Could not download article page via Playwright: {str(pw_err)}")
                finally:
                    await browser.close()

        if not html:
            raise Exception("Could not download article page: Page content is empty.")

        # Check domain-specific selectors first for cleaner text extraction
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        soup = BeautifulSoup(html, "html.parser")
        
        body_content = ""
        if "ft.com" in domain:
            # Targeted FT selectors for article body
            article_body = soup.select_one("div.article-body, div.article__content, [data-trackable='article-body']")
            if article_body:
                paragraphs = article_body.find_all("p")
                body_content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        elif "reuters.com" in domain:
            # Targeted Reuters selectors
            article_body = soup.select_one("article, div.article-body__content, div[class*='article-body']")
            if article_body:
                paragraphs = article_body.find_all("p")
                body_content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        # Fallback 1: Trafilatura generic extractor
        if not body_content:
            try:
                downloaded = trafilatura.extract(html, include_links=True, include_images=True, output_format="txt")
            except Exception as e:
                logger.warning(f"Trafilatura extraction warning: {str(e)}")
                downloaded = None
            body_content = downloaded or ""

        # Fallback 2: General BeautifulSoup paragraphs
        if not body_content:
            paragraphs = soup.find_all("p")
            body_content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        # Check if paywall is blocking content access (before cleaning noise)
        body_lower = body_content.lower()
        paywall_indicators = [
            "subscribe to unlock", "try unlimited access", "become a member", "pricing packages",
            "choose your subscription", "register to read", "standard digital", "rs100 for 4 weeks",
            "rs4499 per month", "unlock this article"
        ]
        if any(ind in body_lower for ind in paywall_indicators):
            logger.warning(f"Paywall blocked content detected for URL: {url}")
            raise ScrapingException("Subscription paywall detected. Access to full article content is restricted.")

        # Filter out boilerplate, promo, cookie and copyright noise
        body_content = self._clean_content(body_content)

        try:
            metadata = trafilatura.extract_metadata(html)
        except Exception:
            metadata = None
        
        title = ""
        author = ""
        published_date = ""
        
        if metadata:
            title = metadata.title or ""
            author = metadata.author or ""
            published_date = metadata.date or ""

        if not title:
            h1 = soup.find("h1")
            title = h1.get_text().strip() if h1 else soup.title.string.strip() if soup.title else "Untitled Article"

        if not author:
            author_meta = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
            if author_meta:
                author = author_meta.get("content", "").strip()

        if not published_date:
            date_meta = soup.find("meta", attrs={"name": "publish-date"}) or soup.find("meta", attrs={"property": "article:published_time"})
            if date_meta:
                published_date = date_meta.get("content", "").strip()

        images_info = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            full_src = urllib.parse.urljoin(url, src)
            alt = img.get("alt", "").strip()
            
            caption = ""
            parent = img.parent
            if parent:
                figcaption = parent.find("figcaption")
                if figcaption:
                    caption = figcaption.get_text().strip()
            
            images_info.append({
                "url": full_src,
                "alt": alt,
                "caption": caption
            })

        return {
            "title": title,
            "author": author,
            "published_date": published_date,
            "source": domain,
            "url": url,
            "content": body_content,
            "images": images_info[:5]
        }

    async def download_image(self, url: str) -> Optional[Image.Image]:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=self.headers) as client:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return Image.open(io.BytesIO(response.content))
            except Exception as e:
                logger.warning(f"Could not download image {url}: {str(e)}")
        return None


# --- 3. EXPLAINABILITY SERVICE ---

class ExplainabilityService:
    def extract_drivers(self, text: str, sentiment: SentimentLabel) -> List[str]:
        drivers = []
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        
        positive_patterns = ["increase", "growth", "subsidy", "profit", "gain", "expand", "boost", "partnership", "success", "innovate"]
        negative_patterns = ["decrease", "decline", "loss", "headwind", "cut", "tighten", "drop", "regulation", "emission penalty", "fine"]
        
        patterns = positive_patterns if sentiment == SentimentLabel.POSITIVE else negative_patterns
        if sentiment == SentimentLabel.NEUTRAL:
            patterns = positive_patterns + negative_patterns
            
        for sent in sentences:
            sent_lower = sent.lower()
            if any(pat in sent_lower for pat in patterns):
                if len(sent) > 20 and len(sent) < 120:
                    drivers.append(sent)
                elif len(sent) >= 120:
                    drivers.append(sent[:117] + "...")
            if len(drivers) >= 4:
                break
                
        if not drivers:
            if sentiment == SentimentLabel.POSITIVE:
                drivers = ["Favorable market conditions noted.", "Strategic growth or expansion indicators present."]
            elif sentiment == SentimentLabel.NEGATIVE:
                drivers = ["Regulative pressure or financial headwinds mentioned.", "Reduction in capacity or target shortfall noted."]
            else:
                drivers = ["Balanced view of current industry developments.", "No significant deviation from baseline performance."]
                
        return drivers[:4]


# --- 4. PIPELINE ORCHESTRATOR ---

class NlpPipelineOrchestrator:
    def __init__(
        self,
        article_repo: ArticleRepository,
        sentiment_repo: SentimentRepository,
        provider_factory: ProviderFactory
    ):
        self.article_repo = article_repo
        self.sentiment_repo = sentiment_repo
        self.providers = provider_factory
        self.scraper = ScraperService()
        self.explainability = ExplainabilityService()

    async def run_url_pipeline(
        self,
        url: str,
        include_ocr: bool,
        include_summary: bool,
        include_entities: bool,
        include_topics: bool,
        include_sentiment_drivers: bool
    ) -> SentimentAnalysisResponseSchema:
        start_time = time.time()
        logger.info(f"Executing URL Sentiment Analysis Pipeline for: {url}")

        # Check cache
        cached_article = self.article_repo.get_by_url(url)
        if cached_article:
            logger.info(f"Found cached article analysis for URL: {url}")
            return self.map_db_to_schema(cached_article, start_time)

        # 1. Scrape Webpage
        scraped_data = await self.scraper.scrape_url(url)
        
        # 2. OCR processing
        ocr_texts = []
        images_processed = 0
        success_ocr = True
        
        if include_ocr and settings.ENABLE_OCR and scraped_data["images"]:
            ocr_provider = self.providers.get_ocr_provider()
            for img_info in scraped_data["images"]:
                img_obj = await self.scraper.download_image(img_info["url"])
                if img_obj:
                    res = ocr_provider.run_ocr(img_obj)
                    if res["success"]:
                        images_processed += 1
                        if res["text"].strip():
                            ocr_texts.append(res["text"].strip())
                    else:
                        success_ocr = False
        
        combined_ocr_text = "\n".join(ocr_texts) if ocr_texts else ""
        merged_body = scraped_data["content"]
        if combined_ocr_text:
            merged_body += f"\n\n[Extracted Image Text]: {combined_ocr_text}"

        # 3. Create & Save Article
        article = Article(
            title=scraped_data["title"],
            author=scraped_data["author"],
            source=scraped_data["source"],
            published_date=scraped_data["published_date"],
            url=url,
            content=merged_body,
            images_found=len(scraped_data["images"]),
            images_processed=images_processed,
            ocr_text=combined_ocr_text,
            ocr_success=success_ocr
        )
        self.article_repo.create(article)

        # 4. Run core analysis
        response_data = await self._run_core_nlp(
            article=article,
            include_summary=include_summary,
            include_entities=include_entities,
            include_topics=include_topics,
            include_sentiment_drivers=include_sentiment_drivers,
            start_time=start_time
        )
        return response_data

    async def _run_core_nlp(
        self,
        article: Article,
        include_summary: bool,
        include_entities: bool,
        include_topics: bool,
        include_sentiment_drivers: bool,
        start_time: float
    ) -> SentimentAnalysisResponseSchema:
        
        # Load Providers
        sentiment_provider = self.providers.get_sentiment_provider()
        ner_provider = self.providers.get_ner_provider()
        summarizer_provider = self.providers.get_summarizer_provider()

        # 1. Overall Sentiment
        label, confidence, reasoning = sentiment_provider.analyze_overall(article.content)
        
        # 2. Summary
        summary = ""
        if include_summary and settings.ENABLE_SUMMARY:
            summary = summarizer_provider.summarize(article.content)

        # 3. NER
        entities = []
        if include_entities and settings.ENABLE_NER:
            entities = ner_provider.extract_entities(article.content)

        # 4. Sentence segmentation (using spaCy helper in summarizer/NER)
        nlp_helper = self.providers.get_ner_provider()._get_nlp()
        doc = nlp_helper(article.content[:3000]) # Segment first 3000 chars for ABSA speed
        sentences = [sent.text.strip() for sent in doc.sents]

        # 5. Aspect Sentiment Analysis
        aspects_results = []
        if settings.ENABLE_ASPECT_SENTIMENT:
            for aspect in settings.ALLOWED_ASPECTS:
                a_label, a_conf, a_reason = sentiment_provider.analyze_aspect(article.content, aspect, sentences)
                if "No direct evidence" not in a_reason:
                    aspects_results.append({
                        "aspect": aspect,
                        "sentiment": a_label,
                        "confidence": a_conf,
                        "reason": a_reason
                    })
        
        if not aspects_results:
            aspects_results.append({
                "aspect": "Energy General",
                "sentiment": label,
                "confidence": confidence,
                "reason": "Evaluated general industry sentiment as no specific aspect met threshold criteria."
            })

        # 6. Sentiment drivers
        drivers = []
        if include_sentiment_drivers:
            drivers = self.explainability.extract_drivers(article.content, label)

        # 7. Topic Detection
        topics = []
        if include_topics:
            topics = [asp["aspect"] for asp in aspects_results]
            for ent in entities[:3]:
                if ent["label"] == "Organization" and ent["text"] not in topics:
                    topics.append(ent["text"])
            if not topics:
                topics = ["Industry Updates", "Market Analysis"]

        # 8. Save all analysis elements
        self.sentiment_repo.save_analysis(
            article_id=article.id,
            overall={"label": label.value, "confidence": confidence, "reasoning": reasoning},
            aspects=aspects_results,
            entities=entities,
            topics=topics,
            drivers=drivers
        )

        processing_time = int((time.time() - start_time) * 1000)
        
        return SentimentAnalysisResponseSchema(
            article=ArticleMetadataSchema(
                title=article.title,
                author=article.author,
                source=article.source,
                published_date=article.published_date,
                url=article.url
            ),
            overall_sentiment=OverallSentimentSchema(
                label=label,
                confidence=confidence,
                reasoning=reasoning
            ),
            summary=summary,
            topics=topics,
            entities=[EntitySchema(text=e["text"], label=e["label"]) for e in entities],
            aspect_sentiment=[
                AspectSentimentSchema(
                    aspect=a["aspect"],
                    sentiment=SentimentLabel(a["sentiment"]),
                    confidence=a["confidence"],
                    reason=a["reason"]
                ) for a in aspects_results
            ],
            sentiment_drivers=drivers,
            ocr=OCRMetadataSchema(
                images_found=article.images_found,
                images_processed=article.images_processed,
                ocr_text=[article.ocr_text] if article.ocr_text else [],
                success=article.ocr_success
            ),
            metadata=ExecutionMetadataSchema(
                processing_time_ms=processing_time,
                model_name=sentiment_provider.model_name,
                model_version="1.0.0",
                timestamp=get_ist_time()
            )
        )

    def map_db_to_schema(self, article: Article, start_time: float) -> SentimentAnalysisResponseSchema:
        overall = article.overall_sentiment
        aspects = article.aspect_sentiments
        entities = article.entities
        topics = [t.name for t in article.topics]
        drivers = [d.driver_text for d in article.drivers]

        processing_time = int((time.time() - start_time) * 1000)
        
        return SentimentAnalysisResponseSchema(
            article=ArticleMetadataSchema(
                title=article.title,
                author=article.author,
                source=article.source,
                published_date=article.published_date,
                url=article.url
            ),
            overall_sentiment=OverallSentimentSchema(
                label=SentimentLabel(overall.label) if overall else SentimentLabel.NEUTRAL,
                confidence=overall.confidence if overall else 1.0,
                reasoning=overall.reasoning if overall else "Cached result loading"
            ),
            summary=article.content[:300] + "...",
            topics=topics,
            entities=[EntitySchema(text=e.text, label=e.label) for e in entities],
            aspect_sentiment=[
                AspectSentimentSchema(
                    aspect=a.aspect,
                    sentiment=SentimentLabel(a.sentiment),
                    confidence=a.confidence,
                    reason=a.reason
                ) for a in aspects
            ],
            sentiment_drivers=drivers,
            ocr=OCRMetadataSchema(
                images_found=article.images_found,
                images_processed=article.images_processed,
                ocr_text=[article.ocr_text] if article.ocr_text else [],
                success=article.ocr_success
            ),
            metadata=ExecutionMetadataSchema(
                processing_time_ms=processing_time,
                model_name=settings.MODEL_NAME,
                model_version="1.0.0",
                timestamp=get_ist_time()
            )
        )
