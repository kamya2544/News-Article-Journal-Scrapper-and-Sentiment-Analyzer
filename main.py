import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn
from sentiment_analysis.core import settings

if __name__ == "__main__":
    # Disable reload on Windows to prevent Uvicorn from forcing the SelectorEventLoop,
    # which raises NotImplementedError for Playwright's subprocesses.
    reload_enabled = sys.platform != "win32"
    uvicorn.run(
        "sentiment_analysis.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=reload_enabled
    )
