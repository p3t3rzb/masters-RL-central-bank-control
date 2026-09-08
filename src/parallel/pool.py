"""Process pools that do not outlive the parent that started them.

Every parallel path in this project has the same shape: a pool of spawned
workers, each holding an expensive private object -- a ``pysolve`` model, a torch
stack, a ground-truth environment -- and chewing a core for minutes at a time on
one unit of work. That shape orphans processes in three distinct ways, and a
stock :class:`~concurrent.futures.ProcessPoolExecutor` closes none of them:

* **The parent is interrupted.** Its context manager exits through
  ``shutdown(wait=True)`` and *without* cancelling, so a ``Ctrl-C`` twenty groups
  into a thousand does not stop the run at all -- it goes on to work through
  every task still queued, which is the whole remainder. :func:`terminate` drops
  the queue and stops the workers instead.
* **The parent is asked to stop.** ``SIGTERM`` on its default disposition kills
  the parent where it stands, so no teardown runs and every worker is left
  behind. :func:`terminating_on_sigterm` turns it into an ordinary unwind.
* **The parent dies outright** -- a ``SIGKILL``, or the shell a backgrounded run
  was launched from going away. A spawned worker blocks on a queue whose write
  end it holds itself, so the parent's death never reaches it as an end-of-file:
  it waits forever, one core each, and no teardown in the parent can help because
  the parent no longer exists. Only the worker can notice, which is what the
  watchdog :func:`arm_worker` starts in it is for.

:func:`managed_pool` is all three guards at once and is what a ``with`` block
should use. :func:`guarded_pool` is the pool alone, for the pools that outlive
any one block and are closed by whichever object owns them -- those must call
:func:`terminate` to close, and carry only the watchdog against a killed parent.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

#: How often a worker checks that its parent is still there. Short enough that an
#: abandoned worker gives up its core in seconds, long enough to cost nothing.
PARENT_POLL_SECONDS = 5.0

#: Grace a terminated worker gets to unwind before it is killed outright. Its
#: results are already in the parent's hands, so this is only about letting a
#: half-written file finish.
TERMINATE_GRACE_SECONDS = 5.0


def _watch_parent(parent_pid: int) -> None:
    """Exit this worker once ``parent_pid`` is no longer its parent.

    Polling :func:`os.getppid` is the portable way to notice a parent that died
    without a teardown -- macOS has no ``PR_SET_PDEATHSIG`` -- and
    :func:`os._exit` is deliberate rather than a raised exception, since an
    orphan has nothing left to report and nobody to flush it to.
    """
    while os.getppid() == parent_pid:
        time.sleep(PARENT_POLL_SECONDS)
    os._exit(1)


def _arm_worker(
    parent_pid: int,
    initializer: Callable[..., None] | None,
    initargs: Sequence[object],
) -> None:
    """Pool initialiser: guard this worker's lifetime, then run the real initialiser.

    Wraps rather than replaces the caller's initialiser, so a pool keeps whatever
    per-worker state it builds and gains the guards on top of it.
    """
    # Ctrl-C reaches every process in the group at once. Leaving it to the parent
    # alone keeps the shutdown ordered -- workers stop when they are told to,
    # rather than each racing to die in the middle of writing its work out.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    if initializer is not None:
        initializer(*initargs)


def guarded_pool(
    max_workers: int,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: Sequence[object] = (),
) -> ProcessPoolExecutor:
    """A spawned pool whose workers stop themselves if this process disappears.

    The pool is otherwise an ordinary executor and the caller owns it: it must be
    closed with :func:`terminate`, and a ``with`` block wants :func:`managed_pool`
    instead, which adds the guards a block can carry.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        # Spawn rather than fork, explicitly: a parent here holds torch and a
        # solver mid-state, neither of which survives forking safely.
        mp_context=mp.get_context("spawn"),
        initializer=_arm_worker,
        initargs=(os.getpid(), initializer, tuple(initargs)),
    )


def terminate(pool: ProcessPoolExecutor) -> None:
    """Drop every queued task and stop the workers still running one.

    The teardown to close a pool with, in place of ``shutdown()``: that waits for
    the whole queue to drain, which on an interrupt means generating the entire
    remainder of the work before it will exit.
    """
    # shutdown() drops the executor's handles on its workers (and with wait=False
    # does not wait for them either), so they have to be collected before it.
    processes = list((pool._processes or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            process.kill()


@contextmanager
def terminating_on_sigterm() -> Iterator[None]:
    """Turn a ``SIGTERM`` into an ordinary unwind for the duration, then restore.

    So that ``kill`` on a long backgrounded run tears its pool down the way an
    interrupt does, rather than killing the parent where it stands and leaving
    the workers to notice on their own. Restores the previous disposition on the
    way out, so importing a module that uses this does not change how the process
    handles signals.

    Off the main thread there is no disposition to set, and this does nothing
    rather than raising: a pool opened from a worker thread should still get the
    guards a worker thread *can* carry, and the watchdog is the one that matters.
    """

    def raise_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"terminated by signal {signum}")

    try:
        previous = signal.signal(signal.SIGTERM, raise_interrupt)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def managed_pool(
    max_workers: int,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: Sequence[object] = (),
) -> Iterator[ProcessPoolExecutor]:
    """A :func:`guarded_pool` that leaves nothing behind, however the block ends.

    A clean exit shuts down gracefully, letting the workers finish what is queued
    and stop by themselves. Anything else -- an interrupt, a ``SIGTERM``, a task
    that raised -- goes out through :func:`terminate` instead, so no worker
    outlives the block it was opened for.
    """
    pool = guarded_pool(max_workers, initializer=initializer, initargs=initargs)
    with terminating_on_sigterm():
        try:
            yield pool
        except BaseException:
            terminate(pool)
            raise
        pool.shutdown(wait=True)
