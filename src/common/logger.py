"""Logger đơn giản dùng rich."""
from rich.console import Console
from rich.logging import RichHandler
import logging

console = Console()


def get_logger(name: str = "har") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(RichHandler(console=console, rich_tracebacks=True))
        logger.setLevel(logging.INFO)
    return logger
