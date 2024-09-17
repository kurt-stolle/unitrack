"""
Simple system to debug tracking modules via process output messages
"""

from __future__ import annotations

import functools

__all__ = ["check_debug_enabled"]


@functools.cache
def check_debug_enabled():
    """
    Check whether debugging is enabled by reading the environment
    variable ``UNITRACK_DEBUG``.
    """
    from unipercept.config.env import get_env

    return get_env(bool, "UNITRACK_DEBUG", default=False)
