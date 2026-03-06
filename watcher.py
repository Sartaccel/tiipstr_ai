import time
import os
import logging

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ingest_utils import ingest_pdf


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
PDF_DIR = "data"


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# PDF Event Handler
# ---------------------------------------------------------
class PDFWatcherHandler(FileSystemEventHandler):
    """
    Handles filesystem events for newly added PDF files.
    """

    def on_created(self, event):

        if event.is_directory:
            return

        if event.src_path.lower().endswith(".pdf"):

            logger.info(f"📄 New PDF detected: {event.src_path}")

            try:
                ingest_pdf(event.src_path)
                logger.info(f"✅ Successfully indexed: {event.src_path}")

            except Exception as e:
                logger.error(f"❌ Failed to process {event.src_path} | {e}")


# ---------------------------------------------------------
# Watcher Service
# ---------------------------------------------------------
def start_pdf_watcher():

    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)

    logger.info(f"👀 Watching for new PDFs in: {PDF_DIR}")

    event_handler = PDFWatcherHandler()
    observer = Observer()

    observer.schedule(
        event_handler,
        path=PDF_DIR,
        recursive=False
    )

    observer.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("🛑 Watcher stopped by user")
        observer.stop()

    observer.join()


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    start_pdf_watcher()