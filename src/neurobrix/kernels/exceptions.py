"""Exceptions pour Kernels Operator - ZERO FALLBACK."""


class KernelNotFoundError(Exception):
    """
    Raised when no kernel is found for the requested configuration.
    
    This is a ZERO FALLBACK error - no automatic fallback is attempted.
    The error message contains actionable information to fix the issue.
    """
    pass


class KernelRegistrationError(Exception):
    """Raised when kernel registration fails (e.g., duplicate)."""
    pass