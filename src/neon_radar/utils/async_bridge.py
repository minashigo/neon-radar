"""Async ↔ Qt bridge.

PySide6's event loop and asyncio's event loop are separate. We solve
this by hosting the asyncio loop inside a regular QThread. The UI
(main thread) calls :meth:`AsyncWorker.submit` from any thread, the
work runs in the worker's loop, and results are delivered back to the
UI via Qt signals — which Qt automatically marshals across threads.

Why this and not ``qasync``
---------------------------
* Zero extra dependency.
* The Qt main thread stays purely UI; it never has to know about
  asyncio primitives.
* Multiple services can share one :class:`AsyncWorker`, or each can
  have its own — both patterns are supported.
* The future returned by :meth:`submit` is a
  :class:`concurrent.futures.Future`, which plays nicely with
  ``add_done_callback``.

Design notes
------------
* :class:`AsyncWorker` is a :class:`QThread` subclass. Its
  :meth:`run` method creates a new event loop and runs it forever.
  Stopping the loop is a normal :meth:`QThread.quit` flow.
* The worker's loop is **private** to this thread. No other code
  should call :func:`asyncio.get_event_loop` and expect to see it.
* Tasks that are still running when :meth:`stop` is called are
  cancelled and awaited, so resources don't leak.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

from PySide6.QtCore import QThread

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Coroutine

T = TypeVar("T")


class AsyncWorker(QThread):
    """A :class:`QThread` that hosts an asyncio event loop.

    Lifecycle::

        worker = AsyncWorker()
        worker.start()                       # spawn thread, start loop
        future = worker.submit(some_coro())  # schedule work
        future.add_done_callback(handler)    # runs in worker thread
        worker.stop()                        # graceful shutdown
    """

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The asyncio loop if running, else ``None``.

        Useful for advanced use cases (e.g. scheduling a callback
        directly). Most callers should use :meth:`submit`.
        """
        return self._loop

    @property
    def is_running(self) -> bool:
        return self._loop is not None and self._loop.is_running()

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # Drain pending tasks so resources (e.g. httpx clients)
            # close cleanly.
            try:
                pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                self._loop.close()
                self._loop = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Schedule a coroutine on this worker's loop.

        Returns a :class:`concurrent.futures.Future`. The future's
        ``done_callback`` runs in the **worker thread**, so callbacks
        should be quick and not perform UI work directly — emit a Qt
        signal instead and let Qt deliver it to the main thread.

        Raises
        ------
        RuntimeError
            If the worker is not running yet or has been stopped.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "AsyncWorker is not running. Call start() and wait briefly "
                "before submit()."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self, timeout_ms: int = 5_000) -> None:
        """Stop the asyncio loop and wait for the thread to finish."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.quit()
        if not self.wait(timeout_ms):
            # Force termination as a last resort. Pending coroutines
            # were already cancelled in run()'s finally block.
            self.terminate()
            self.wait(timeout_ms)
