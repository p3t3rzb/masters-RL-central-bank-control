"""Shared worker-process plumbing: pools that clean up after themselves.

One module, :mod:`~parallel.pool`, holding the teardown every parallel path in
this project needs and none of them get for free -- an interrupt that actually
stops the work, a ``SIGTERM`` that unwinds, and workers that notice a parent that
died without a teardown at all. See that module's docstring for what each guard
is for; :func:`~parallel.pool.managed_pool` is the one to reach for.
"""

from parallel.pool import (
    guarded_pool,
    managed_pool,
    terminate,
    terminating_on_sigterm,
)

__all__ = [
    "guarded_pool",
    "managed_pool",
    "terminate",
    "terminating_on_sigterm",
]
