"""Strongly Connected Component (SCC) detection for loop identification."""

from __future__ import annotations

import networkx as nx

from armature.discovery.dfg import DFG

# Type alias for loop context: maps activity to SCC index (or None if not in loop)
LoopContext = dict[str, int | None]


def find_sccs(dfg: DFG) -> LoopContext:
    """Find strongly connected components (loops) in the DFG.

    Uses NetworkX's Tarjan's algorithm to detect SCCs. Activities in the same
    SCC can reach each other, forming a loop. Single-activity SCCs are only
    included if they have a self-loop.

    Args:
        dfg: Directly-follows graph

    Returns:
        LoopContext mapping each activity to its SCC index (or None if not in SCC)
    """
    # Build NetworkX directed graph from DFG edges
    graph = nx.DiGraph()

    # Add all activities as nodes
    graph.add_nodes_from(dfg.activities)

    # Add edges from DFG
    for source, targets in dfg.edges.items():
        for target in targets:
            graph.add_edge(source, target)

    # Find all strongly connected components
    sccs = list(nx.strongly_connected_components(graph))

    # Build loop context: only include SCCs with size > 1 OR self-loops
    loop_ctx: LoopContext = {}
    scc_index = 0

    for scc in sccs:
        # Check if this is a multi-activity SCC or has a self-loop
        if len(scc) > 1:
            # Multi-activity SCC is a loop
            for activity in scc:
                loop_ctx[activity] = scc_index
            scc_index += 1
        elif len(scc) == 1:
            # Single activity - check for self-loop
            activity = next(iter(scc))
            if activity in dfg.edges and activity in dfg.edges[activity]:
                # Self-loop detected
                loop_ctx[activity] = scc_index
                scc_index += 1

    # Activities not in any SCC (not in loop_ctx) are implicitly None
    # But we don't need to explicitly set them to None
    return loop_ctx
