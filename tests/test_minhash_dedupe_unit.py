import importlib.util
from pathlib import Path

import pytest
import daft
from daft import col


from hypothesis import given, settings
from hypothesis.strategies import text, lists, integers, booleans, floats, composite

pytestmark = []


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


@pytest.fixture(scope="module")
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


@given(text())
@settings(max_examples=50)
def test_remove_http_headers_hypothesis(mod, input_str):
    fn = mod.remove_http_headers  
    result = fn(input_str)
    assert isinstance(result, str)
    if "\r\n\r\n" in input_str:
        assert result == input_str.split("\r\n\r\n", 1)[1]
    else:
        assert result == ""


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
    # Exercise get_block_idx via a small pipeline
    df_blocks = df_blocks.with_column("idx", mod.get_block_idx(col("blocks")))
    pd = df_blocks.to_pydict()
    assert pd["idx"][0] == list(range(len(pd["blocks"][0])))


def test_extract_blocks_more_tags(mod):
    html = (
        '<html><head>'
        '<meta name="description" content="desc">'
        '<meta property="og:title" content="ogt">'
        '<meta property="og:description" content="ogd">'
        '</head><body>'
        '<img alt="pic" />'
        '<figure><figcaption>caption</figcaption></figure>'
        '</body></html>'
    )
    blocks = mod.extract_blocks(html)
    assert any("caption" in b for b in blocks)
    # Also ensure no empty strings are included
    assert all(len(b) > 0 for b in blocks)


def test_extract_blocks_strips_script_style_noscript(mod):
    html = (
        '<html><head><style>.x{color:red}</style><script>var a=1;</script></head>'
        '<body><noscript>nope</noscript><p>ok</p></body></html>'
    )
    blocks = mod.extract_blocks(html)
    assert any("ok" in b for b in blocks)
    assert all("nope" not in b for b in blocks)


def test_sig_and_diff_helpers(mod):
    a = daft.from_pydict({"u": [1,2], "v": [2,3]})
    b = daft.from_pydict({"u": [2,3], "v": [1,2]})
    Asig = mod.sig(a)
    Bsig = mod.sig(b)
    AminusB, BminusA = mod.diff(a, b)
    assert isinstance(Asig, set) and isinstance(Bsig, set)
    assert isinstance(AminusB, set) and isinstance(BminusA, set)


def test_remove_http_headers_direct(mod):
    body = mod.remove_http_headers("A:1\r\n\r\nhello")
    assert body == "hello"
    assert mod.remove_http_headers(None) == ""
    assert mod.remove_http_headers("") == ""


def test_get_block_idx_direct(mod):
    idx = mod.get_block_idx(["a", "b", "c"])
    assert idx == [0, 1, 2]


@given(text())
@settings(max_examples=50)
def test_extract_blocks_hypothesis(mod, html):
    fn = mod.extract_blocks  
    blocks = fn(html)
    assert isinstance(blocks, list)
    assert all(isinstance(b, str) for b in blocks)


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


@given(floats(0.1, 0.9), integers(16, 256))
@settings(max_examples=50, deadline=None)
def test_optimal_param_properties(mod, threshold, num_perm):
    B, R = mod.optimal_param(threshold, num_perm)
    assert B * R <= num_perm
    assert B >= 1 and R >= 1


def test_optimal_param_weighting(mod):
    # Different weights should still return valid (B,R)
    B1, R1 = mod.optimal_param(0.7, 64, 0.9, 0.1)
    B2, R2 = mod.optimal_param(0.7, 64, 0.1, 0.9)
    assert B1 * R1 <= 64 and B2 * R2 <= 64


# --- Normalization tests ---------------------------------------------------------

@given(text(), booleans(), booleans(), booleans(), booleans())
@settings(max_examples=100, deadline=None)
def test_normalize_idempotent(pipeline, text, remove_punct, lowercase, nfd_unicode, white_space):
    df = daft.from_pydict({"block_text": [text]})
    norm1 = pipeline.normalize(df, remove_punct, lowercase, nfd_unicode, white_space)
    norm2 = pipeline.normalize(norm1, remove_punct, lowercase, nfd_unicode, white_space)
    assert norm1.to_pydict()["content_normalized"] == norm2.to_pydict()["content_normalized"]

@pytest.mark.parametrize("flags", [
    (True, True, True, True),
    (False, False, False, False),
    (True, False, True, False),
])
def test_normalize_specific_cases(pipeline, flags):
    text = "Hello, World! cafés  \t\n"
    df = daft.from_pydict({"block_text": [text]})
    norm = pipeline.normalize(df, *flags).to_pydict()["content_normalized"][0]
    if all(flags):
        assert norm == "hello world cafe\u0301s"
    elif not any(flags):
        assert norm == text


# --- MinHash tests ---------------------------------------------------------------

@given(text(min_size=10), integers(16, 64), integers(3,7), integers(1,1000))
@settings(max_examples=50)
def test_minhash_properties(pipeline, text, num_perm, ngram_size, seed):
    # Include the required index column as expected by pipeline.minhash
    df = daft.from_pydict({pipeline.index_col: ["id"], "content_normalized": [text]})
    mh = pipeline.minhash(df, num_perm, ngram_size, seed, "xxhash")
    hashes = mh.to_pydict()["min_hashes"][0]
    assert len(hashes) == num_perm
    assert all(isinstance(h, int) for h in hashes)


# --- LSH Banding and Grouping tests ----------------------------------------------

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


@composite
def hashes_and_R(draw):
    R = draw(integers(2, 8))
    k = draw(integers(1, 50))
    length = R * k
    min_hashes = draw(lists(integers(0, 1000), min_size=length, max_size=length))
    return min_hashes, R


@given(hashes_and_R())
def test_lsh_banding_properties(pipeline, data):
    min_hashes, R = data
    B = len(min_hashes) // R
    df = daft.from_pydict({"node_id": [0], "min_hashes": [min_hashes]})
    banded = pipeline.lsh_banding(df, R, B)
    assert banded.count_rows() <= B


def test_band_generation_wrapper(pipeline):
    R, B = 2, 2
    df = daft.from_pydict({pipeline.index_col: [1], "min_hashes": [[1,2,3,4]]})
    banded = pipeline.band_generation(df, R, B)
    out = banded.to_pydict()
    assert {"node_id", "bands", "band_idx"}.issubset(set(out.keys()))


def test_prep_node_id_index_map(pipeline):
    df = daft.from_pydict({"block_id": ["a", "b"], "min_hashes": [[1], [2]]})
    df_node, id_map = pipeline.prep_node_id_index_map(df)
    assert "node_id" in df_node.to_pydict()
    assert set(id_map.to_pydict()["block_id"]) == {"a", "b"}


# --- Connected Components tests --------------------------------------------------

def test_generate_edges_from_nodes(pipeline):
    # Two clusters: [1,2,3] and [4,5]
    grouped = daft.from_pydict({
        "nodes": [
            [1, 2, 3],
            [4, 5],
        ]
    })
    # Build edges using current _build_edges which yields columns u, v
    edges = pipeline._build_edges(grouped)
    out = edges.to_pydict()
    left = out["u"]
    right = out["v"]
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
    # Invariants: u != v
    assert all(u != v for u, v in zip(out["u"], out["v"]))


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


def test_check_canonical_set_equality_true_and_false(pipeline):
    # Equal after canonicalization (orientation ignored)
    a = daft.from_pydict({"u": [1, 2], "v": [2, 3]})
    b = daft.from_pydict({"u": [2, 3], "v": [1, 2]})
    assert pipeline.check_canonical_set_equality(a, b) is True

    # Different canonical sets => not equal
    b2 = daft.from_pydict({"u": [1, 3], "v": [2, 4]})
    assert pipeline.check_canonical_set_equality(a, b2) is False


@pytest.mark.parametrize("algorithm", ["alternating", "two_phase"])
def test_connected_components_algorithms(pipeline, algorithm):
    # Complex graph with cycles and singletons
    nodes_groups = [[1,2,3,4], [5,6], [7]]  # components: 1-4, 5-6, 7
    df = daft.from_pydict({"band_idx": list(range(len(nodes_groups))), "bands": [0]*len(nodes_groups), "nodes": nodes_groups})
    assigns = pipeline.connected_components_2(df, algorithm=algorithm)
    pdict = assigns.to_pydict()
    mapping = dict(zip(pdict["u"], pdict["rep"]))
    assert mapping[1] == mapping[2] == mapping[3] == mapping[4] == min(1,2,3,4)
    assert mapping[5] == mapping[6] == min(5,6)
    # Isolated nodes may be omitted from assignments in current API
    assert 7 not in mapping or mapping[7] == 7

@pytest.mark.parametrize("algorithm", ["alternating", "two_phase"])
def test_connected_components_break_on_empty_group(pipeline, algorithm):
    # Singletons only -> no edges -> immediate convergence and break
    df = daft.from_pydict({"band_idx": [0,1], "bands": [0,1], "nodes": [[7], [8]]})
    assigns = pipeline.connected_components_2(df, algorithm=algorithm)
    assert assigns.count_rows() == 0

@pytest.mark.skipif(importlib.util.find_spec("igraph") is None, reason="igraph not installed")
def test_connected_components_with_igraph_validate(pipeline):
    nodes_groups = [[1,2],[3,4]]
    df = daft.from_pydict({"band_idx": [0,1], "bands": [0,1], "nodes": nodes_groups})
    assigns = pipeline.connected_components_2(df, algorithm="alternating", igraph_validate=True)
    pdict = assigns.to_pydict()
    assert len(pdict["u"]) > 0


def test_global_min_label_propagation(pipeline):
    edges = daft.from_pydict({"u": [1,2,3], "v": [2,3,4]})
    assigns = daft.from_pydict({"u": [1,2,3,4], "rep": [1,2,3,4]})  # initial disconnected
    final = pipeline.global_min_label_propagation(edges, assigns)
    pdict = final.to_pydict()
    assert all(r == 1 for r in pdict["rep"])


def test_construct_assignments_singletons(pipeline):
    b = daft.from_pydict({"u": [], "v": []})  # no edges
    # Current API does not support empty edge tables; expect an error
    with pytest.raises(Exception):
        pipeline.construct_assignments(b)


# --- Merge and Output tests ------------------------------------------------------

def test_merge_results(pipeline):
    df = daft.from_pydict({"block_id": ["a", "b", "c"], "block_text": ["txt1", "txt2", "txt3"]})
    assigns = daft.from_pydict({"u": [0,1,2], "rep": [0,0,2]})
    id_map = daft.from_pydict({"block_id": ["a","b","c"], "node_id": [0,1,2]})
    results = pipeline.merge_results(df, assigns, id_map)
    kept = results.to_pydict()["block_id"]
    assert sorted(kept) == ["a", "c"]  # reps "a" (for a,b) and "c"


# --- Preprocess integration test (tiny) -----------------------------------------

def test_preprocess_filters_and_extracts(mod, pipeline):
    # Validate header removal and block extraction using current helpers
    html_body = "<main><p>Alpha</p><p>Beta</p></main>"
    warc = f"X:1\r\nY:2\r\n\r\n{html_body}"
    df = daft.from_pydict(
        {
            "WARC-Identified-Payload-Type": ["text/html", "application/json"],
            # Provide strings to match remove_http_headers signature
            "warc_content": [warc, "{\"k\":1}"],
        }
    )

    df_html = df.where(col("WARC-Identified-Payload-Type") == "text/html")
    df_content = df_html.with_column("content_raw", mod.remove_http_headers(col("warc_content")))
    df_blocks = df_content.with_column("blocks", mod.extract_blocks(col("content_raw")))
    blocks = df_blocks.to_pydict()["blocks"][0]
    assert any("Alpha" in b for b in blocks) and any("Beta" in b for b in blocks)


# --- Optional S3 integration for preprocess -------------------------------------

@pytest.mark.integration
def test_preprocess_common_crawl_integration_env_gated(mod):
    # Only run if credentials and sample URI are provided
    from dotenv import load_dotenv
    load_dotenv()
    import os
    uri = os.getenv("COMMONCRAWL_SAMPLE_URI")
    key = os.getenv("AWS_ACCESS_KEY_ID")
    sec = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not (uri and key and sec):
        pytest.skip("Missing env for S3 integration test")

    # Ensure limited row processing
    df = mod.preprocess_common_crawl_html(uri=uri, row_limit=5, index_col="block_id", content_col="block_text")
    pdf = df.to_pydict()
    assert len(pdf["block_id"]) >= 0  # sanity: call succeeded


# --- Full Pipeline Integration (small scale) -------------------------------------

def test_full_pipeline_end_to_end(mod, pipeline):
    input_df = daft.from_pydict({
        "block_id": ["1", "2", "3"],
        "block_text": ["hello world", "hello world", "unique text"]
    })
    results = pipeline(input_df)
    pdict = results.to_pydict()
    assert len(pdict["block_id"]) == 2  # deduped to one "hello" and the unique
    assert set(pdict["block_text"]) == {"hello world", "unique text"}


# --- Partitioning tests ----------------------------------------------------------

def test_partitioned_save_no_ray(mod):
    df = daft.from_pydict({"x": [1]})
    mod.partitioned_save("dummy", df, 100, 10)  # should fall back to simple write

@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="Ray not installed")
def test_partitioned_save_with_ray(mod):
    import ray
    ray.init(ignore_reinit_error=True)
    df = daft.from_pydict({"x": list(range(1000))})
    mod.partitioned_save("dummy", df, 100, 10)  # test repartition path
    ray.shutdown()

# Additional coverage for small helpers and edge cases

def test_ee_sig_diff_and_canonicalize_symmetrize(mod, pipeline):
    # ee constructs a struct; use via small_star path implicitly covered. Here test canonicalize+symmetrize
    edges = daft.from_pydict({"u": [3, 2], "v": [1, 3]})
    can = pipeline.canonicalize(edges).to_pydict()
    pairs = set(zip(can["u"], can["v"]))
    assert pairs == {(1, 3), (2, 3)}

    sym = pipeline.symmetrize(edges).to_pydict()
    assert set(sym.keys()) == {"u", "v"}
    assert len(sym["u"]) == 4  # two edges both directions

    # sig and diff utilities
    s = pipeline.canonicalize(edges)
    s2 = pipeline.canonicalize(edges.select(col("v").alias("u"), col("u").alias("v")))
    assert pipeline.check_canonical_set_equality(s, s2)


def test_pairs_equal(pipeline):
    a = daft.from_pydict({"u": [1,2], "rep": [1,1]})
    b = daft.from_pydict({"u": [1,2], "rep": [1,1]})
    assert pipeline._pairs_equal(a, b)
    c = daft.from_pydict({"u": [1,2], "rep": [1,2]})
    assert not pipeline._pairs_equal(a, c)


def test_merge_results_isolated_and_self_rep(pipeline):
    df = daft.from_pydict({"block_id": ["a", "b"], "block_text": ["A", "B"]})
    assigns = daft.from_pydict({"u": [0], "rep": [0]})
    id_map = daft.from_pydict({"block_id": ["a", "b"], "node_id": [0, 1]})
    results = pipeline.merge_results(df, assigns, id_map).to_pydict()
    # Keep "a" (self-representative) and keep isolated "b" (no assignment)
    assert sorted(results["block_id"]) == ["a", "b"]
