import hashlib
import urllib.parse
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import httpx
from loguru import logger
import pytz
from sqlalchemy.orm import Session
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sentiment_analysis.db import SessionLocal, Article, ScraperConfig, ScraperLog
from sentiment_analysis.services import NlpPipelineOrchestrator, ScraperService, ArticleRepository, SentimentRepository
from sentiment_analysis.providers import get_provider_factory
from sentiment_analysis.core import get_ist_time, settings

# --- 1. SCRAPER ORCHESTRATOR ---

class ScraperOrchestrator:
    def __init__(self, db: Session, pipeline: NlpPipelineOrchestrator):
        self.db = db
        self.pipeline = pipeline
        self.scraper_service = ScraperService()

    async def run_scraper(self, config_id: int) -> ScraperLog:
        log_entry = ScraperLog(
            config_id=config_id,
            status="running",
            articles_scraped=0,
            articles_updated=0,
            started_at=get_ist_time()
        )
        self.db.add(log_entry)
        self.db.commit()
        self.db.refresh(log_entry)

        config = self.db.query(ScraperConfig).filter(ScraperConfig.id == config_id).first()
        if not config:
            log_entry.status = "failed"
            log_entry.error_message = f"Config with ID {config_id} not found."
            log_entry.completed_at = get_ist_time()
            self.db.commit()
            return log_entry

        try:
            logger.info(f"Running scraper for config '{config.name}' targeting listing: {config.scrape_url}")
            html = None
            use_playwright = False
            fallback_reason = ""
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=self.scraper_service.headers) as client:
                try:
                    response = await client.get(config.scrape_url)
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
                        elif response.status_code != 200:
                            raise Exception(f"Listing page responded with status {response.status_code}")
                except Exception as e:
                    logger.warning(f"HTTP request error for listing page: {str(e)}. Falling back to Playwright.")
                    fallback_reason = f"HTTP request exception: {str(e)}"
                    use_playwright = True

            if use_playwright:
                logger.info(f"Falling back to Playwright for listing page: {config.scrape_url} due to: {fallback_reason}")
                from playwright.async_api import async_playwright
                from playwright_stealth import Stealth
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    context = await browser.new_context(
                        user_agent=self.scraper_service.headers["User-Agent"],
                        viewport={"width": 1280, "height": 800}
                    )
                    page = await context.new_page()
                    await Stealth().apply_stealth_async(page)
                    try:
                        await page.goto(config.scrape_url, wait_until="domcontentloaded", timeout=30000)
                        html = await page.content()
                    except Exception as pw_err:
                        logger.error(f"Playwright listing page navigation failed: {str(pw_err)}")
                        try:
                            html = await page.content()
                        except Exception:
                            raise Exception(f"Could not download listing page via Playwright: {str(pw_err)}")
                    finally:
                        await browser.close()

            if not html:
                raise Exception("Could not download listing page: Content is empty.")

            soup = BeautifulSoup(html, "html.parser")
            article_urls = []
            
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                full_url = urllib.parse.urljoin(config.scrape_url, href)
                parsed_url = urllib.parse.urlparse(full_url)
                path = parsed_url.path.strip("/")
                
                if config.base_url not in full_url:
                    continue
                
                exclusions = [
                    "signin", "login", "register", "signup", "privacy", "terms", "cookies",
                    "search", "tag", "category", "author", "about", "contact", "help", "faq",
                    "facebook", "twitter", "linkedin", "instagram", "youtube", "feed", "rss",
                    "newsletter", "subscribe", "advertising", "careers"
                ]
                if any(ex in path.lower() for ex in exclusions):
                    continue
                
                parts = [p for p in path.split("/") if p]
                if len(parts) < 2:
                    continue
                
                if full_url not in article_urls:
                    article_urls.append(full_url)

            if settings.MAX_ARTICLES_PER_FEED > 0:
                article_urls = article_urls[:settings.MAX_ARTICLES_PER_FEED]
            logger.info(f"Found {len(article_urls)} candidate article URLs: {article_urls}")

            scraped_count = 0
            updated_count = 0

            for url in article_urls:
                try:
                    # 1. Check if URL already exists in DB to save scraping time & cost
                    existing_article_url = self.db.query(Article).filter(Article.url == url).first()
                    if existing_article_url:
                        logger.info(f"Article URL already exists in database. Skipping scrape: {url}")
                        continue

                    # Scrape content
                    scraped_data = await self.scraper_service.scrape_url(url)
                    content = scraped_data["content"]
                    if not content:
                        continue

                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    # 2. Check if content hash already exists in DB (different URL syndication)
                    existing_article_hash = self.db.query(Article).filter(Article.content_hash == content_hash).first()
                    if existing_article_hash:
                        logger.info(f"Article content hash already exists in database (ID {existing_article_hash.id}). Skipping: {url}")
                        continue

                    scraped_count += 1

                    await self.pipeline.run_url_pipeline(
                        url=url,
                        include_ocr=True,
                        include_summary=True,
                        include_entities=True,
                        include_topics=True,
                        include_sentiment_drivers=True
                    )

                    new_article = self.db.query(Article).filter(Article.url == url).first()
                    if new_article:
                        new_article.content_hash = content_hash
                        self.db.commit()

                except Exception as ex:
                    logger.error(f"Error scraping or analyzing article {url}: {str(ex)}")

            log_entry.status = "success"
            log_entry.articles_scraped = scraped_count
            log_entry.articles_updated = updated_count

        except Exception as e:
            logger.error(f"Scraper run failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            log_entry.status = "failed"
            log_entry.error_message = str(e)

        log_entry.completed_at = get_ist_time()
        self.db.commit()
        return log_entry


# --- 2. SCHEDULER ORCHESTRATION ---

IST = pytz.timezone("Asia/Kolkata")
scheduler = AsyncIOScheduler(timezone=IST)

async def execute_scheduled_job(config_id: int):
    logger.info(f"Triggering scheduled scraper job for config ID {config_id}")
    db: Session = SessionLocal()
    try:
        article_repo = ArticleRepository(db)
        sentiment_repo = SentimentRepository(db)
        pipeline = NlpPipelineOrchestrator(
            article_repo=article_repo,
            sentiment_repo=sentiment_repo,
            provider_factory=get_provider_factory()
        )
        orchestrator = ScraperOrchestrator(db, pipeline)
        await orchestrator.run_scraper(config_id)
    except Exception as e:
        logger.error(f"Failed to execute scheduled job for config ID {config_id}: {str(e)}")
    finally:
        db.close()

def sync_jobs():
    logger.info("Syncing scheduler jobs with database configurations...")
    db: Session = SessionLocal()
    try:
        for job in scheduler.get_jobs():
            job.remove()

        configs = db.query(ScraperConfig).filter(ScraperConfig.is_active == True).all()
        for config in configs:
            try:
                scheduler.add_job(
                    execute_scheduled_job,
                    CronTrigger.from_crontab(config.schedule_cron, timezone=IST),
                    id=f"scraper_{config.id}",
                    args=[config.id],
                    replace_existing=True
                )
                logger.info(f"Scheduled scraper job '{config.name}' (ID {config.id}) with cron: {config.schedule_cron}")
            except Exception as e:
                logger.error(f"Failed to schedule job for config '{config.name}': {str(e)}")
    finally:
        db.close()

def update_job_schedule(config: ScraperConfig):
    job_id = f"scraper_{config.id}"
    try:
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

        if config.is_active:
            scheduler.add_job(
                execute_scheduled_job,
                CronTrigger.from_crontab(config.schedule_cron, timezone=IST),
                id=job_id,
                args=[config.id],
                replace_existing=True
            )
            logger.info(f"Updated schedule for job '{config.name}' (ID {config.id}): {config.schedule_cron}")
        else:
            logger.info(f"Deactivated job for config '{config.name}' (ID {config.id})")
    except Exception as e:
        logger.error(f"Failed to update job '{config.name}' schedule: {str(e)}")

def remove_job(config_id: int):
    job_id = f"scraper_{config_id}"
    try:
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
            logger.info(f"Removed job {job_id} from scheduler")
    except Exception as e:
        logger.error(f"Failed to remove job {job_id}: {str(e)}")
