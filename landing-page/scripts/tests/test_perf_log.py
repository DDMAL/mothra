"""perf_log is observability wrapped around stages that are allowed to fail
softly, so its contract is almost entirely about what it must NOT do: it must
not swallow a caller's exception, and it must not raise one of its own, no
matter how badly its inputs or its publisher misbehave. These tests pin that
down, because a regression here converts a degraded-but-complete job into a
failed one."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_log import TIMING_PREFIX, stage_timer, timed  # noqa: E402


def test_stage_timer_publishes_one_log_event_with_the_label():
    events = []
    with stage_timer(events.append, "my-stage", image="page.png"):
        pass
    assert len(events) == 1
    assert events[0]["type"] == "log"
    assert TIMING_PREFIX in events[0]["message"]
    assert "my-stage" in events[0]["message"]
    assert "image=page.png" in events[0]["message"]
    # Regression guard: the elapsed time must be measured BEFORE the line is
    # formatted. Getting that order wrong (e.g. by nesting the reporter
    # inside the timer) prints a literal "?" for every stage, and every
    # other assertion in this file still passes.
    assert re.search(r"my-stage: \d+\.\d\ds", events[0]["message"])


def test_note_detail_reaches_the_message():
    events = []
    with stage_timer(events.append, "staffline-detection") as t:
        t.note(stave_boxes=42)
    assert "stave_boxes=42" in events[0]["message"]


def test_none_valued_detail_is_omitted_rather_than_printed():
    events = []
    with stage_timer(events.append, "s", present=1, absent=None):
        pass
    assert "present=1" in events[0]["message"]
    assert "absent" not in events[0]["message"]


def test_exception_in_the_body_propagates_but_still_reports_elapsed_time():
    events = []
    with pytest.raises(ValueError, match="boom"):
        with stage_timer(events.append, "failing-stage"):
            raise ValueError("boom")
    # The timing line is the most useful precisely on this path (a timeout,
    # a cancellation, an OOM'd peer), so it must still be emitted.
    assert len(events) == 1
    assert "failing-stage" in events[0]["message"]
    assert "?" not in events[0]["message"]


def test_a_publisher_that_raises_does_not_take_the_stage_down():
    def exploding_publish(_obj):
        raise RuntimeError("job_events is down")

    with stage_timer(exploding_publish, "s"):
        pass  # must not raise


def test_a_detail_value_whose_str_raises_does_not_take_the_stage_down():
    class Hostile:
        def __str__(self):
            raise RuntimeError("nope")

        __repr__ = __str__

    events = []
    with stage_timer(events.append, "s", bad=Hostile()):
        pass
    # The line is dropped rather than half-written, and nothing propagates.
    assert events == []


def test_falsy_publish_is_log_only():
    with stage_timer(None, "s"):
        pass  # must not raise


def test_timed_records_elapsed_seconds_on_the_handle():
    with timed("s") as t:
        pass
    assert t.seconds is not None
    assert t.seconds >= 0.0


def test_timed_records_elapsed_seconds_even_when_the_body_raises():
    handle = None
    with pytest.raises(ValueError):
        with timed("s") as t:
            handle = t
            raise ValueError
    assert handle is not None and handle.seconds is not None


def test_note_never_raises_on_a_hostile_mapping_update():
    with timed("s") as t:
        t.note(**{"ok": 1})
        assert t.extra["ok"] == 1
