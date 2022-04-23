import warnings
from cvlization.logging.logging import configure_logging


warnings.filterwarnings("ignore", category=DeprecationWarning)
configure_logging()
