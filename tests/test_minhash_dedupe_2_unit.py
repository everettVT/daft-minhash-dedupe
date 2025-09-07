import importlib.util
from pathlib import Path

import pytest
import daft
from daft import col
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "workload" / "minhash_dedupe_2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("minhash_dedupe_2", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture()
def pipeline(mod):
    return mod.MinHashDedupePipeline(
        output_uri=str(PROJECT_ROOT / "data" / "output"),
        checkpoint_uri=str(PROJECT_ROOT / "data" / "checkpoint"),
    )


def test_optimal_param_known_example(mod):
    B, R = mod.optimal_param(0.7, 256)
    assert (B, R) == (25, 10)


def test_band_generation_and_grouping(mod, pipeline):
    R, B = 2, 3
    min_hashes = [1, 2, 3, 4, 5, 6]
    df = daft.from_pydict({pipeline.index_col: [0, 1], "min_hashes": [min_hashes, min_hashes]})
    banded = pipeline.lsh_banding(df.with_column("node_id", col(pipeline.index_col)), R, B)
    out = banded.to_pydict()
    assert len(out["band_idx"]) == B
    for nodes in out["nodes"]:
        assert sorted(nodes) == [0, 1]


def test_connected_components_from_bands(mod, pipeline):
    # Two clusters via bands
    grouped = daft.from_pydict({
        "band_idx": [0, 1],
        "bands": [[1, 2], [3, 4]],
        "nodes": [[1, 2, 3], [4, 5]],
    })
    assigns = pipeline.connected_components_2(grouped)
    pd = assigns.to_pydict()
    mp = dict(zip(pd["u"], pd["rep"]))
    for n in (1, 2, 3):
        assert mp[n] == min(1, 2, 3)
    for n in (4, 5):
        assert mp[n] == min(4, 5)


def test_connected_components_vs_networkx_edges(mod):
    # Build undirected simple graph (drop self-loops, dedup)
    edges = {(1, 2), (2, 3), (4, 5)}
    df_edges = daft.from_pydict({"u": [1, 2, 4], "v": [2, 3, 5]})
    from workload.connected_components import ConnectedComponents

    cc = ConnectedComponents()
    assigns = cc.compute_from_edges(df_edges)
    pd = assigns.to_pydict()
    ours = {}
    for u, r in zip(pd["u"], pd["rep"]):
        ours.setdefault(r, set()).add(u)
    ours = {frozenset(s) for s in ours.values()}

    G = nx.Graph()
    G.add_edges_from(edges)
    nx_comps = {frozenset(c) for c in nx.connected_components(G)}
    assert ours == nx_comps


