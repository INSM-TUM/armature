"""Tests for XES event log parsing."""

import gzip
from datetime import datetime, timezone
from pathlib import Path

from armature.discovery.models import Event, Trace
from armature.discovery.xes_parser import parse_xes

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_empty_log():
    """Empty log should return empty list."""
    traces = parse_xes(FIXTURES_DIR / "empty.xes")
    assert traces == []


def test_single_trace_single_event():
    """Single trace with single event."""
    # Create minimal XES file
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="Case1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2024-01-01T10:00:00.000+00:00"/>
    </event>
  </trace>
</log>
"""
    path = FIXTURES_DIR / "single.xes"
    path.write_text(xes_content)

    traces = parse_xes(path)

    assert len(traces) == 1
    assert traces[0].case_id == "Case1"
    assert len(traces[0].events) == 1
    assert traces[0].events[0].activity == "A"
    assert traces[0].events[0].timestamp == datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def test_single_trace_multiple_events():
    """Single trace with multiple events maintains order."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="Case1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2024-01-01T10:00:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <date key="time:timestamp" value="2024-01-01T11:00:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
      <date key="time:timestamp" value="2024-01-01T12:00:00.000+00:00"/>
    </event>
  </trace>
</log>
"""
    path = FIXTURES_DIR / "ordered.xes"
    path.write_text(xes_content)

    traces = parse_xes(path)

    assert len(traces) == 1
    trace = traces[0]
    assert trace.case_id == "Case1"
    assert len(trace.events) == 3
    assert [e.activity for e in trace.events] == ["A", "B", "C"]


def test_multiple_traces():
    """Multiple traces parsed correctly."""
    traces = parse_xes(FIXTURES_DIR / "simple.xes")

    assert len(traces) == 2

    # First trace
    assert traces[0].case_id == "Case1"
    assert len(traces[0].events) == 2
    assert traces[0].events[0].activity == "A"
    assert traces[0].events[1].activity == "B"

    # Second trace
    assert traces[1].case_id == "Case2"
    assert len(traces[1].events) == 2
    assert traces[1].events[0].activity == "A"
    assert traces[1].events[1].activity == "C"


def test_missing_timestamp():
    """Event with missing timestamp should have None."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="Case1"/>
    <event>
      <string key="concept:name" value="A"/>
    </event>
  </trace>
</log>
"""
    path = FIXTURES_DIR / "no_timestamp.xes"
    path.write_text(xes_content)

    traces = parse_xes(path)

    assert len(traces) == 1
    assert traces[0].events[0].timestamp is None


def test_missing_case_id():
    """Trace with missing case_id gets generated ID."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <event>
      <string key="concept:name" value="A"/>
    </event>
  </trace>
  <trace>
    <event>
      <string key="concept:name" value="B"/>
    </event>
  </trace>
</log>
"""
    path = FIXTURES_DIR / "no_case_id.xes"
    path.write_text(xes_content)

    traces = parse_xes(path)

    assert len(traces) == 2
    assert traces[0].case_id == "trace_0"
    assert traces[1].case_id == "trace_1"


def test_path_as_string():
    """Parser accepts string path."""
    traces = parse_xes(str(FIXTURES_DIR / "empty.xes"))
    assert traces == []


def test_event_model_validation():
    """Event model validates correctly."""
    event = Event(activity="A")
    assert event.activity == "A"
    assert event.timestamp is None

    ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    event_with_ts = Event(activity="B", timestamp=ts)
    assert event_with_ts.timestamp == ts


def test_trace_model_validation():
    """Trace model validates correctly."""
    events = [
        Event(activity="A"),
        Event(activity="B", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
    ]
    trace = Trace(case_id="Case1", events=events)

    assert trace.case_id == "Case1"
    assert len(trace.events) == 2
    assert trace.events[0].activity == "A"


def test_parse_xes_gzipped():
    """Parser handles gzipped XES files."""
    xes_content = """<?xml version="1.0" encoding="UTF-8"?>
<log>
  <trace>
    <string key="concept:name" value="Case1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2024-01-01T10:00:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <date key="time:timestamp" value="2024-01-01T11:00:00.000+00:00"/>
    </event>
  </trace>
  <trace>
    <string key="concept:name" value="Case2"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2024-01-01T12:00:00.000+00:00"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
      <date key="time:timestamp" value="2024-01-01T13:00:00.000+00:00"/>
    </event>
  </trace>
</log>
"""
    # Write compressed XES file
    path = FIXTURES_DIR / "compressed.xes.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(xes_content)

    # Parse gzipped file
    traces = parse_xes(path)

    assert len(traces) == 2

    # First trace
    assert traces[0].case_id == "Case1"
    assert len(traces[0].events) == 2
    assert traces[0].events[0].activity == "A"
    assert traces[0].events[1].activity == "B"

    # Second trace
    assert traces[1].case_id == "Case2"
    assert len(traces[1].events) == 2
    assert traces[1].events[0].activity == "A"
    assert traces[1].events[1].activity == "C"
