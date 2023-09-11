import logging
import logging.config

from dotenv import load_dotenv

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)
load_dotenv()  # take environment variables from .env.
