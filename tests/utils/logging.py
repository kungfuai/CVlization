import logging
import sys
from logging.config import dictConfig


DEFAULT_LOGGING = {"version": 1, "disable_existing_loggers": False}


def configure_logging(level=logging.INFO):
    """Configure console logging for tests."""
    dictConfig(DEFAULT_LOGGING)
    try:
        import coloredlogs

        coloredlogs.install(
            level=level,
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
        )
    except ImportError:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
            stream=sys.stdout,
        )
