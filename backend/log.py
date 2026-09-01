"""Logging setup shared by every CLI entry point."""
import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)
