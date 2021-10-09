import warnings
from cvlization.config.logging import configure_logging


warnings.filterwarnings("ignore", category=DeprecationWarning)
configure_logging()
