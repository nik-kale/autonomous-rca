"""Investigation Graph visualization and analysis utilities.

Provides functions to:
- Render the Investigation Graph as a color-coded matplotlib figure
- Extract the root-cause reasoning path
- Export the graph as a clean JSON structure
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from src.state import (
    CAUSES,
    CONTRADICTS,
    DEPENDS_ON,
    EVIDENCE,
    GENERATED_FROM,
    HYPOTHESIS,
    PROBLEM,
    REJECTED,
    ROOT_CAUSE,
    SUPPORTS,
)

# Node type → colour mapping
_NODE_COLORS: dict[str, str] = {
    PROBLEM: "#e74c3c",      # red
    HYPOTHESIS: "#3498db",   # blue
    EVIDENCE: "#1abc9c",     # teal
    REJECTED: "#95a5a6",     # gray
    ROOT_CAUSE: "#2ecc71",   # green
}

# Edge type → style mapping
_EDGE_STYLES: dict[str, dict[str, Any]] = {
    GENERATED_FROM: {"color": "#7f8c8d", "style": "dashed"},
    SUPPORTS: {"color": "#27ae60", "style": "solid"},
    CONTRADICTS: {"color": "#e74c3c", "style": "dotted"},
    CAUSES: {"color": "#e67e22", "style": "solid"},
    DEPENDS_ON: {"color": "#9b59b6", "style": "dashed"},
}


def _build_nx_graph(
    nodes: list[dict], edges: list[dict]
) -> nx.DiGraph:
    """Convert raw node/edge lists into a NetworkX directed graph."""
    G = nx.DiGraph()

    for node in nodes:
        G.add_node(
            node["id"],
            type=node.get("type", "unknown"),
            text=node.get("text", ""),
            iteration=node.get("iteration", 0),
            confidence=node.get("confidence"),
        )

    for edge in edges:
        G.add_edge(
            edge["from_id"],
            edge["to_id"],
            type=edge.get("type", "unknown"),
            iteration=edge.get("iteration", 0),
        )

    return G


def plot_investigation_graph(
    nodes: list[dict],
    edges: list[dict],
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
) -> None:
    """Render the Investigation Graph as a matplotlib figure.

    Nodes are colored by type; edges are styled by relationship type.
    Iteration numbers are shown as subscript labels.
    """
    current_backend = matplotlib.get_backend()
    is_interactive = current_backend not in ("agg", "Agg", "")
    if not is_interactive and save_path:
        matplotlib.use("Agg")
    G = _build_nx_graph(nodes, edges)

    if not G.nodes:
        print("Empty graph — nothing to visualize.")
        return

    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw nodes
    for node_type, color in _NODE_COLORS.items():
        type_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        if type_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=type_nodes, node_color=color,
                node_size=800, alpha=0.9, ax=ax,
            )

    # Node labels: short text + iteration
    labels = {}
    for n, d in G.nodes(data=True):
        text = d.get("text", n)
        short = text[:40] + "..." if len(text) > 40 else text
        labels[n] = f"{short}\n(iter {d.get('iteration', '?')})"

    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Draw edges
    for edge_type, style in _EDGE_STYLES.items():
        type_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("type") == edge_type
        ]
        if type_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=type_edges,
                edge_color=style["color"],
                style=style["style"],
                arrows=True,
                arrowsize=15,
                ax=ax,
            )

    # Edge labels
    edge_labels = {(u, v): d.get("type", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, font_size=6, font_color="#555555", ax=ax
    )

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markersize=10, label=ntype.replace("_", " ").title())
        for ntype, color in _NODE_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title("Investigation Graph", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def extract_root_cause_path(
    nodes: list[dict], edges: list[dict]
) -> list[dict]:
    """Extract the reasoning path from the problem node to the root cause.

    Performs a BFS from the problem node following ``generated_from``,
    ``supports``, ``causes``, and ``depends_on`` edges to reach the
    root-cause node.  Returns the ordered list of nodes along that path.
    """
    G = _build_nx_graph(nodes, edges)

    # Find the problem and root-cause nodes
    problem_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == PROBLEM]
    rc_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == ROOT_CAUSE]

    if not problem_nodes or not rc_nodes:
        return []

    start = problem_nodes[0]

    # BFS for shortest path to any root-cause node
    for rc in rc_nodes:
        try:
            path_ids = nx.shortest_path(G, start, rc)
            return [
                {"id": nid, **G.nodes[nid]} for nid in path_ids
            ]
        except nx.NetworkXNoPath:
            continue

    # Fallback: return all root-cause nodes even without a connected path
    return [{"id": nid, **G.nodes[nid]} for nid in rc_nodes]


def export_graph_json(nodes: list[dict], edges: list[dict]) -> dict:
    """Export the Investigation Graph as a clean JSON-serializable dict."""
    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": _count_by_key(nodes, "type"),
            "edge_types": _count_by_key(edges, "type"),
        },
    }


def _count_by_key(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts
