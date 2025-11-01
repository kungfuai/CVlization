import warnings
from tests.utils.logging import configure_logging


warnings.filterwarnings("ignore", category=DeprecationWarning)
configure_logging()
