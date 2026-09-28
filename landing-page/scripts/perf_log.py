"""Stage-timing helpers shared by the Celery tasks and the service bridges.

Everything here is *observability only*: it measures how long a stage took
and emits one line describing it. Nothing in this module may change control
flow, and nothing in it may raise -- the stages it wraps
(paco-classifier, text-finding, staffline detection, pitch finding) are all
allowed to fail softly and report the failure themselves, so a timer that
raised on its way out would convert a degraded-but-complete job into a
failed one. Every public entry point here is therefore wrapped so that a
formatting bug, a publisher that throws, or a clock that goes backwards
costs a missing log line and nothing more.

Two entry points, same measurement, different destinations:

- `stage_timer(publish, label, **extra)` -- for code that already has a
  job-event `publish` callable (tasks_predict.py, tasks_encode.py). Emits a
  `{"type": "log", "message": "[timing] ..."}` event, which
  `GET /api/jobs/{id}/stream` relays and `ProcessingPage.tsx` renders
  verbatim alongside the stage's own log lines -- so these land in the job
  log the user already reads, with no frontend change.
- `timed(label, **extra)` -- for code with no publisher (ic_api.py's
  request handlers, paco-classifier-service). Records the elapsed time on
  the context object for the caller to put wherever it belongs (a response
  body, an SSE frame) and logs it at INFO.

Both yield a small mutable object, so a stage that only learns its own
"size" partway through (how many boxes it iterated, how many patches it
ran) can attach that after the fact:

    with stage_timer(publish, "staffline-detection", image=name) as t:
        n = run_it()
        t.note(boxes=n)
"""
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# The marker every emitted line carries. Grep-able in a downloaded job log
# and in `kubectl logs`, and distinct enough not to collide with the
# stages' own free-text messages.
TIMING_PREFIX = "[timing]"


class StageTiming:
    """Handle yielded by `stage_timer`/`timed` -- collects the label, the
    elapsed seconds, and any key/value detail the stage attaches."""

    __slots__ = ("label", "extra", "seconds")

    def __init__(self, label, extra):
        self.label = label
        self.extra = dict(extra)
        # Set on context exit. Stays None if the clock somehow fails, which
        # `format()` renders as "?" rather than crashing on arithmetic.
        self.seconds = None

    def note(self, **extra):
        """Attach detail discovered while the stage ran (counts, sizes,
        which branch was taken). Never raises: a stage calling this is
        mid-flight, and losing a log detail must not cost it its work."""
        try:
            self.extra.update(extra)
        except Exception:  # noqa: BLE001 - see the module docstring
            pass

    def format(self):
        secs = "?" if self.seconds is None else f"{self.seconds:.2f}s"
        detail = ", ".join(f"{k}={v}" for k, v in self.extra.items() if v is not None)
        return f"{TIMING_PREFIX} {self.label}: {secs}" + (f" ({detail})" if detail else "")


@contextmanager
def _measure(label, extra, on_finish):
    """Shared body of `stage_timer`/`timed`: time the block, record the
    elapsed seconds on the handle, and only THEN hand it to `on_finish` for
    reporting.

    That order is the whole reason this is one function rather than two
    nested context managers. Nesting reports-inside-times gets it backwards:
    the inner `finally` runs first, so the reporter would format a handle
    whose `seconds` had not been set yet and every line would read "?".

    The timing and the report both live in a `finally`, so a stage that
    raises still says how long it ran before it did -- which is exactly the
    case where the number matters most (a timeout, a cancellation, an OOM'd
    peer). The exception itself propagates untouched; this module never
    swallows a caller's error, only its own.
    """
    handle = StageTiming(label, extra)
    start = time.perf_counter()
    try:
        yield handle
    finally:
        try:
            handle.seconds = time.perf_counter() - start
        except Exception:  # noqa: BLE001 - see the module docstring
            pass
        try:
            on_finish(handle)
        except Exception:  # noqa: BLE001 - see the module docstring
            pass


@contextmanager
def stage_timer(publish, label, **extra):
    """Time a block and publish one `[timing]` job-event log line for it.

    `publish` is the job-event callable the task already has; a falsy
    `publish` makes this log-only, so a helper shared between a task and a
    non-task caller doesn't need two code paths.
    """
    def _report(handle):
        message = _safe_format(handle)
        if message is None:
            return
        logger.info(message)
        if publish:
            publish({"type": "log", "message": message})

    with _measure(label, extra, _report) as handle:
        yield handle


@contextmanager
def timed(label, **extra):
    """Time a block, log it, and leave the number on the handle for the
    caller to forward (a response field, an SSE frame). Same guarantees as
    `stage_timer`, minus the job-event publish."""
    def _report(handle):
        message = _safe_format(handle)
        if message is not None:
            logger.info(message)

    with _measure(label, extra, _report) as handle:
        yield handle


def _safe_format(handle):
    """`StageTiming.format()` touches caller-supplied values via f-string,
    so a `__str__` that raises (or a NaN duration) must not take the stage
    down with it. Returns None when the line can't be built at all."""
    try:
        return handle.format()
    except Exception:  # noqa: BLE001 - see the module docstring
        return None
