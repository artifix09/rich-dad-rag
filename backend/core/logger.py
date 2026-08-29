import sys
from loguru import logger
from backend.core.config import settings


def setup_logger() -> None:

    # Remove the default loguru handler
    logger.remove()

    # --- Console Handler ---
    logger.add(
        sys.stdout,
        level="DEBUG" if settings.ENVIRONMENT == "development" else "INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # --- File Handler ---
    logger.add(
        "logs/app.log",
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
    )


setup_logger()