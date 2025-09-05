import importlib.util
from pathlib import Path

import pytest
import daft
import networkx as nx
from hypothesis import given, strategies as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "workload" / "minhash_dedupe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("minhash_dedupe", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="module")
def pipeline():
    mod = _load_module()
    return mod.CommonCrawlHtmlMinHashDedupe(
        output_uri=str(PROJECT_ROOT / "data" / "output"),
        checkpoint_uri=str(PROJECT_ROOT / "data" / "checkpoint"),
    )


def test_cc_matches_networkx_deterministic(pipeline):
    # Build a deterministic graph with multiple components and some redundancy
    edges = {
        (1, 2), (2, 3), (1, 3),  # triangle component {1,2,3}
        (4, 5),                   # small component {4,5}
        (6, 7), (7, 8),           # chain {6,7,8}
    }
    left, right = zip(*sorted((min(u, v), max(u, v)) for (u, v) in edges))
    daft_edges = daft.from_pydict({"left_edge": list(left), "right_edge": list(right)})

    ours = pipeline.connected_components(daft_edges).to_pydict()
    ours_map = dict(zip(ours[pipeline.index_col], ours[pipeline.component_col]))

    G = nx.Graph()
    G.add_edges_from(edges)
    nx_components = [frozenset(c) for c in nx.connected_components(G)]

    by_label = {}
    for node, label in ours_map.items():
        by_label.setdefault(label, set()).add(node)
    ours_components = [frozenset(s) for s in by_label.values()]

    assert set(ours_components) == set(nx_components)


@given(
    st.integers(min_value=1, max_value=30).flatmap(
        lambda n: st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
            ),
            min_size=0,
            max_size=200,
        )
    )
)
def test_cc_property_matches_networkx(edges_list, pipeline):
    # Build undirected simple graph (drop self-loops, dedup)
    edges = {(min(u, v), max(u, v)) for (u, v) in edges_list if u != v}
    if not edges:
        return
    left, right = zip(*sorted(edges))
    daft_edges = daft.from_pydict({"left_edge": list(left), "right_edge": list(right)})

    ours = pipeline.connected_components(daft_edges).to_pydict()
    ours_map = dict(zip(ours[pipeline.index_col], ours[pipeline.component_col]))

    G = nx.Graph()
    G.add_edges_from(edges)
    nx_components = [frozenset(c) for c in nx.connected_components(G)]

    by_label = {}
    for node, label in ours_map.items():
        by_label.setdefault(label, set()).add(node)
    ours_components = [frozenset(s) for s in by_label.values()]

    assert set(ours_components) == set(nx_components)
