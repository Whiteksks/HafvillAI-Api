import logging
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

from .app import create_app
from .settings import Settings


def main():
    # Environment variables take precedence over the local .env file.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    settings = Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1,
                proxy_headers=False, log_level="info")


if __name__ == "__main__":
    main()
