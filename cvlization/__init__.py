"""
Copyright (c) 2022 KUNGFU.AI.
All rights reserved.
"""


def configure_logging(level=None):
    """Configure default logging for consumers of the library.

    Call this explicitly if you want cvlization's default logging setup.
    """
    import logging

    if level is None:
        level = logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
        )
    else:
        logging.getLogger().setLevel(level)
