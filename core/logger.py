import os
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
