"""Discovery algorithms (XES parsing, SCC detection, temporal/existential dependencies)."""

from armature.discovery.dfg import DFG, build_dfg
from armature.discovery.discover import discover
from armature.discovery.existential import extract_existential_dependencies
from armature.discovery.models import Event, Trace
from armature.discovery.scc import LoopContext, find_sccs
from armature.discovery.temporal import extract_temporal_dependencies
from armature.discovery.xes_parser import parse_xes

__all__ = [
    "DFG",
    "Event",
    "LoopContext",
    "Trace",
    "build_dfg",
    "discover",
    "extract_existential_dependencies",
    "extract_temporal_dependencies",
    "find_sccs",
    "parse_xes",
]
