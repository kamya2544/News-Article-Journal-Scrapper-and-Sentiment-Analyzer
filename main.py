import asyncio
import sys
import uvicorn
from sentiment_analysis.core import settings

# On Windows, we configure the Proactor event loop to ensure proper asynchronous operations.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == "__main__":
    # We disable reload on Windows because Uvicorn's reload feature forces the SelectorEventLoop,
    # which is incompatible with Playwright's subprocess requirements.
    reload_enabled = sys.platform != "win32"
    uvicorn.run(
        "sentiment_analysis.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=reload_enabled
    )
