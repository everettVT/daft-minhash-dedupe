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
def mod():
    return _load_module()


def test_cc_matches_networkx_deterministic(mod):
    # Build a deterministic graph with multiple components and some redundancy
    pipeline = mod.CommonCrawlHtmlMinHashDedupe(
        output_uri=str(PROJECT_ROOT / "data" / "output"),
        checkpoint_uri=str(PROJECT_ROOT / "data" / "checkpoint"),
    )

    # Prep Edges
    if 
    cc_uri = 
    df_raw = pipeline.load_data(cc_uri, row_limit=1000)
    df_prepped = self.preprocess(df_raw)
    df_norm = pipeline.normalize(df_prepped)
    df_minhash = pipeline.minhash(df_norm)
    B, R = mod.optimal_param(df_minhash)
    df_grouped = pipeline.group_bands(df_minhash, R, B)
    df_grouped = pipeline.checkpoint(df_grouped, "bands", persist_checkpoint=True)

    



    df_pd_edges = df_edges_clean.to_pandas()

# using networkx @YK we can just pick one. IGraph is faster but only has strong/weak components. 
nx_graph = nx.from_pandas_edgelist(df_pd_edges, source="u", target="v")
nx_components = [frozenset(c) for c in nx.connected_components(nx_graph)]


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


import networkx as nx
import igraph as ig

df_pd_edges = df_edges_clean.to_pandas()

# using networkx @YK we can just pick one. IGraph is faster but only has strong/weak components. 
nx_graph = nx.from_pandas_edgelist(df_pd_edges, source="u", target="v")
nx_components = [frozenset(c) for c in nx.connected_components(nx_graph)]

# using igraph
g = ig.Graph.DataFrame(df_pd_edges, directed=False)
strong_components = [frozenset(c) for c in g.connected_components(mode="strong")]
weak_components = [frozenset(c) for c in g.connected_components(mode="weak")]

print(nx_components)
print(strong_components)
print(weak_components)
assert nx_components == strong_components == weak_components