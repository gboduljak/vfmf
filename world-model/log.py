import logging
import os
from pathlib import Path


def create_logger(logging_dir: Path, rank: int) -> logging.Logger:
  """
  Create a logger that writes to a log file and stdout for rank 0 only.
  For all other ranks, the logger is disabled.
  """
  logger = logging.getLogger(f"logger_rank_{rank}")  # unique per rank
  logger.propagate = False  # don't pass logs to parent

  if logger.hasHandlers():
    return logger

  if rank == 0:
    os.makedirs(logging_dir, exist_ok=True)
    logger.setLevel(logging.INFO)

    # StreamHandler (stdout) with color
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        '[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)

    # FileHandler (log file) without color
    file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
  else:
    # Effectively disables this logger
    logger.setLevel(logging.CRITICAL + 1)

  return logger
