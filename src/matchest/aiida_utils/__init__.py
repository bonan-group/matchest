"""
AiiDA related tools
"""

try:
    import aiida  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "The aiida_utils module requires AiiDA dependencies. "
        "Install with: pip install matchest[aiida] or uv pip install matchest[aiida]"
    ) from exc
