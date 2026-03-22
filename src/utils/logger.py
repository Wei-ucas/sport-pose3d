import datetime
import logging
import os
from logging.handlers import QueueHandler, QueueListener
from concurrent_log_handler import ConcurrentRotatingFileHandler
from typing import Optional, Union


def config_root_logger(
    # name: Optional[str] = None,
    log_level: Union[int, str, None] = None,
    log_file_path: Optional[str] = None,
) -> None:
    """
    Configure the root logger with a specific logging level and file path.
    Args:
        log_level (Union[int, str, None]): Logging level (e.g., logging.INFO).
        log_file_path (Optional[str]): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    format_pattern = (
        "[%(asctime)s|%(processName)s|%(name)s|%(filename)s, line:%(lineno)d] "
        "[%(levelname)s] %(message)s"
    )

    if log_file_path is None:
        log_file_path = f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    handlers = [
        ConcurrentRotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ]

    logging.basicConfig(level=log_level, format=format_pattern, handlers=handlers)


class QueueListenerRoot(QueueListener):
    """
    A QueueListener that respects the root logger's level.
    """

    def __init__(self, queue: logging.handlers.QueueHandler):
        super().__init__(queue, *logging.root.handlers, respect_handler_level=True)

    def handle(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.root.level:
            super().handle(record)


def child_process_config(queue: logging.handlers.QueueHandler, log_level: Union[int, str]) -> None:
    """
    Configure logging for child processes.

    Args:
        queue (logging.handlers.QueueHandler): Queue for logging.
        log_level (Union[int, str]): Logging level.
    """
    root = logging.getLogger()
    root.addHandler(QueueHandler(queue))
    root.setLevel(log_level)