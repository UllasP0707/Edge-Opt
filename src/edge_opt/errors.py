"""Domain-specific Edge-Opt exceptions."""


class EdgeOptError(Exception):
    """Base exception for expected Edge-Opt failures."""


class ConfigurationError(EdgeOptError, ValueError):
    """Raised when a model or hardware configuration is invalid."""


class AccuracyBudgetExceeded(EdgeOptError):
    """Raised when an optimized model violates the configured quality bound."""

