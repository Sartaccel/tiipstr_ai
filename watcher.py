import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingest_utils import ingest_pdf

PDF_DIR = "data"

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            ingest_pdf(event.src_path)

if __name__ == "__main__":
    print("👀 Watching for new PDFs...")
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, PDF_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Watcher stopped")

    observer.join()
