from pathlib import Path

import daft
import networkx as nx
import pytest
from hypothesis import given, strategies as st, settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _daft_edges_from_pairs(pairs):
    if not pairs:
        return daft.from_pydict({"u": [], "v": []})
    left, right = zip(*pairs)
    return daft.from_pydict({"u": list(left), "v": list(right)})


def _components_from_assignments(assign_df, u_col="u", rep_col="rep"):
    pd_assign = assign_df.to_pydict()
    nodes = pd_assign[u_col]
    reps = pd_assign[rep_col]
    by_label = {}
    for n, r in zip(nodes, reps):
        by_label.setdefault(r, set()).add(n)
    return {frozenset(s) for s in by_label.values()}


def _nx_components_from_edges(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return {frozenset(c) for c in nx.connected_components(G)}


def test_cc_matches_networkx_small_graph():
    # Triangle + chain + isolated
    edges = {(0, 1), (1, 2), (3, 4), (4, 5)}
    df_edges = _daft_edges_from_pairs(sorted(edges))
    from workload.connected_components import ConnectedComponents

    cc = ConnectedComponents()
    assigns = cc.compute_from_edges(df_edges)

    ours = _components_from_assignments(assigns)
    nx_comps = _nx_components_from_edges(edges)
    assert ours == nx_comps


@given(
    st.integers(min_value=1, max_value=30).flatmap(
        lambda n: st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
            ),
            min_size=0,
            max_size=150,
        )
    )
)
@settings(max_examples=50, deadline=None)
def test_cc_property_matches_networkx(edges_list):
    # Build undirected simple graph (drop self-loops, dedup, canonicalize)
    edges = {(min(u, v), max(u, v)) for (u, v) in edges_list if u != v}
    if not edges:
        return
    df_edges = _daft_edges_from_pairs(sorted(edges))

    from workload.connected_components import ConnectedComponents

    cc = ConnectedComponents()
    assigns = cc.compute_from_edges(df_edges)

    ours = _components_from_assignments(assigns)
    nx_comps = _nx_components_from_edges(edges)
    assert ours == nx_comps


