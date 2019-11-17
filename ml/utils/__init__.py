"""Provides utility methods."""

import logging


class LogMixin:  # pylint: disable=too-few-public-methods
    """Adds a logger property to the applied class."""
    @property
    def _logger(self) -> logging.Logger:
        """Return a logger instance."""
        return logging.getLogger(
            type(self).__name__
        )
