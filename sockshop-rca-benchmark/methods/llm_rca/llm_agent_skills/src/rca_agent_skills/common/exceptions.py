class RCAAgentError(Exception):
    """Base RCA framework error."""


class ConfigurationError(RCAAgentError):
    """Configuration error."""


class DataAccessError(RCAAgentError):
    """Backend access error."""


class BudgetExceededError(RCAAgentError):
    """Follow-up query budget exceeded."""

