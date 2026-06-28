"""Directly-Follows Graph (DFG) construction from event traces."""

from __future__ import annotations

from pydantic import BaseModel, Field

from armature.discovery.models import Trace


class DFG(BaseModel):
    """Directly-Follows Graph data structure.

    Attributes:
        activities: Set of all unique activities in the traces
        edges: Dict mapping source activity -> target activity -> count
        start_activities: Activities that start traces
        end_activities: Activities that end traces
    """

    activities: set[str] = Field(default_factory=set, description="All unique activities")
    edges: dict[str, dict[str, int]] = Field(default_factory=dict, description="Source -> target -> count")
    start_activities: set[str] = Field(default_factory=set, description="Activities starting traces")
    end_activities: set[str] = Field(default_factory=set, description="Activities ending traces")


def build_dfg(traces: list[Trace]) -> DFG:
    """Build a Directly-Follows Graph from traces.

    Args:
        traces: List of process traces containing ordered events

    Returns:
        DFG with activities, edges, start/end activities
    """
    activities: set[str] = set()
    edges: dict[str, dict[str, int]] = {}
    start_activities: set[str] = set()
    end_activities: set[str] = set()

    for trace in traces:
        if not trace.events:
            # Skip empty traces
            continue

        # First event is start activity
        start_activities.add(trace.events[0].activity)
        # Last event is end activity
        end_activities.add(trace.events[-1].activity)

        # Process consecutive pairs to build edges
        for i in range(len(trace.events)):
            activity = trace.events[i].activity
            activities.add(activity)

            # Add edge from current to next activity
            if i < len(trace.events) - 1:
                next_activity = trace.events[i + 1].activity
                if activity not in edges:
                    edges[activity] = {}
                edges[activity][next_activity] = edges[activity].get(next_activity, 0) + 1

    return DFG(
        activities=activities,
        edges=edges,
        start_activities=start_activities,
        end_activities=end_activities,
    )
