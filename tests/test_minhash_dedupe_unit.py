import importlib.util
from pathlib import Path

import pytest
import daft
from daft import col

pytestmark = []


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "workload" / "minhash_dedupe_2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("minhash_dedupe", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture()
def pipeline(mod):
    # Use dummy URIs; they won't be touched in these tests
    return mod.MinHashDedupePipeline(
        output_uri=str(PROJECT_ROOT / "data" / "output"),
        checkpoint_uri=str(PROJECT_ROOT / "data" / "checkpoint"),
    )


# --- Pure function tests ---------------------------------------------------------

def test_remove_http_headers(mod):
    fn = mod.remove_http_headers  

    df = daft.from_pydict(
        {
            "warc_content": [
                "Header-A: x\r\nHeader-B: y\r\n\r\n<body>hi</body>",
                "no-headers-here",
                None,
                "",
            ]
        }
    )
    out = df.with_column("html_content", fn(col("warc_content")))
    out = out.to_pydict()
    assert out["html_content"] == ["<body>hi</body>", "", "", ""]



def test_extract_blocks_basic(mod):
    fn = mod.extract_blocks  
    
    df_html = daft.from_pydict(
        {
            "html": [
                """<html><head><style>.x{color:red}</style><script>var a=1</script></head>
                <body>
                <h1>Title</h1>
                <p>Hello <b>world</b>.</p>
                <noscript>ignore me</noscript>
                </body></html>""",
            ]
        }
    )

    df_blocks = df_html.with_column("blocks", fn(col("html")))
    blocks = df_blocks.to_pydict()["blocks"]
    text_blocks = blocks[0]
    assert isinstance(text_blocks, list)
    assert len(text_blocks) == 2
    assert "Title" in text_blocks
    assert "Hello world ." in text_blocks 
    assert "ignore me" not in text_blocks

    

# --- optimal_param tests ---------------------------------------------------------

def test_optimal_param_known_example(mod):
    # From doc/example for threshold=0.7, num_perm=256
    B, R = mod.optimal_param(0.7, 256)
    assert (B, R) == (25, 10)


def test_optimal_param_constraints_small(mod):
    for threshold in (0.5, 0.7, 0.9):
        B, R = mod.optimal_param(threshold, 64)
        assert isinstance(B, int) and isinstance(R, int)
        assert B >= 1 and R >= 1
        assert B * R <= 64



# --- Band generation/grouping tests ---------------------------------------------

def test_band_generation_and_grouping(mod, pipeline):
    # Create two documents with identical min_hashes so they collide in all bands
    R, B = 2, 3  # R * B must equal len(min_hashes)
    min_hashes = [1, 2, 3, 4, 5, 6]

    df = daft.from_pydict(
        {
            pipeline.index_col: [0, 1],
            "min_hashes": [min_hashes, min_hashes],
        }
    )

    # Use pipeline.lsh_banding with explicit node_id column
    banded = pipeline.lsh_banding(df.with_column("node_id", col(pipeline.index_col)), R, B)
    grouped = banded

    # Expect one group per band (since both docs share identical bands),
    # with nodes list containing both ids [0, 1]
    out = grouped.to_pydict()
    assert len(out["band_idx"]) == B
    for nodes in out["nodes"]:
        assert sorted(nodes) == [0, 1]


# --- Connected components phases -------------------------------------------------

def test_generate_edges_from_nodes(pipeline):
    # Two clusters: [1,2,3] and [4,5]
    grouped = daft.from_pydict({
        "nodes": [
            [1, 2, 3],
            [4, 5],
        ]
    })
    # Re-implement expected edges using pipeline._build_edges semantics
    edges = pipeline._build_edges(grouped)
    out = edges.to_pydict()
    left = out["left_edge"]
    right = out["right_edge"]
    pairs = set(zip(left, right))
    # left_edge is the min in each group
    assert (1, 2) in pairs and (1, 3) in pairs
    assert (4, 5) in pairs
    # no self-loops
    assert all(left_node != right_node for left_node, right_node in pairs)


def test_large_star_phase_invariants(pipeline):
    # Input undirected edges for two components
    edges = daft.from_pydict({
        "u": [1, 2, 4],
        "v": [2, 3, 5],
    })
    a = pipeline._large_star(edges)
    out = a.to_pydict()
    assert set(out.keys()) == {"u", "v"}
    assert len(out["u"]) == len(out["v"]) > 0
    # Invariants: u != v, and kept edges are oriented smaller->larger
    assert all(u != v for u, v in zip(out["u"], out["v"]))
    assert all(u < v for u, v in zip(out["u"], out["v"]))


def test_small_star_phase_invariants(pipeline):
    # Start from the output of large-star phase
    edges = daft.from_pydict({
        "u": [1, 2, 4],
        "v": [2, 3, 5],
    })
    a = pipeline._large_star(edges)
    b = pipeline._small_star(a)
    out = b.to_pydict()
    assert set(out.keys()) == {"u", "v"}
    assert len(out["u"]) == len(out["v"]) > 0
    assert all(u != v for u, v in zip(out["u"], out["v"]))


def test_check_convergence_true_and_false(pipeline):
    # Same u set => converged
    a = daft.from_pydict({"u": [1, 2], "v": [9, 9]})
    b = daft.from_pydict({"u": [1, 2], "v": [8, 7]})
    assert pipeline._check_canonical_set_equality(a, b) is True

    # Different u set => not converged
    b2 = daft.from_pydict({"u": [1, 3], "v": [8, 7]})
    assert pipeline._check_canonical_set_equality(a, b2) is False


# --- Connected components composite ---------------------------------------------

def test_connected_components_two_components(mod, pipeline):
    # Graph: {1-2-3} and {4-5}; include extra edges to ensure stability
    edges = daft.from_pydict(
        {
            "left_edge": [1, 2, 4, 6, 7, 9, 10],
            "right_edge": [2, 3, 5, 7, 8, 10, 11],
        }
    )

    # Convert test edges to bands-like input and run CC2 directly on edges
    assigns = pipeline.connected_components_2(
        daft.from_pydict({
            "band_idx": [0],
            "bands": [[0]],
            "nodes": [[1, 2, 3], [4, 5]][0:1]  # placeholder; not used by build_edges here
        })
    )
    # Alternatively build edges then compute via ConnectedComponents
    df_edges = daft.from_pydict({"u": [1, 2, 4, 6, 7, 9, 10], "v": [2, 3, 5, 7, 8, 10, 11]})
    from workload.connected_components import ConnectedComponents
    cc = ConnectedComponents()
    assignments = cc.compute_from_edges(df_edges)
    pdict = assignments.to_pydict()
    ids = pdict[pipeline.index_col]
    comps = pdict[pipeline.component_col]
    mapping = dict(zip(ids, comps))

    # All nodes observed
    for node in (1, 2, 3, 4, 5):
        assert node in mapping

    # Expect canonical label to be the smallest id in each component
    for node in (1, 2, 3):
        assert mapping[node] == 1
    for node in (4, 5):
        assert mapping[node] == 4


# --- Preprocess integration test (tiny) -----------------------------------------

def test_preprocess_filters_and_extracts(mod, pipeline):
    # Two rows: only the HTML row should remain with extracted content
    html_body = "<main><p>Alpha</p><p>Beta</p></main>"
    warc = f"X:1\r\nY:2\r\n\r\n{html_body}"
    df = daft.from_pydict(
        {
            "WARC-Identified-Payload-Type": ["text/html", "application/json"],
            "warc_content": [warc.encode("utf-8"), b"{\"k\":1}"],
        }
    )

    out = mod.preprocess_common_crawl_html.__wrapped__(df) if hasattr(mod.preprocess_common_crawl_html, "__wrapped__") else pipeline.prep(df)
    pdict = out.to_pydict()

    # Only one row should pass through
    assert len(pdict[pipeline.content_col]) == 1
    text = pdict[pipeline.content_col][0]
    assert "Alpha" in text and "Beta" in text
