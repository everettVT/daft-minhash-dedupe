
import os
from selectolax.parser import HTMLParser
import math
import time

import daft
from daft import col, lit, struct, DataFrame, Expression
from daft.functions import monotonically_increasing_id
from daft.io import IOConfig, S3Config

from scipy.integrate import quad as integrate
from .connected_components import ConnectedComponents


from logging import getLogger, INFO

logger = getLogger(__name__)
logger.setLevel(INFO)


@daft.func()
def remove_http_headers(x: str) -> str:
    if x is None:
        return ""
    if len(x.split("\r\n\r\n")) > 1:
        return x.split("\r\n\r\n")[1]
    return ""

@daft.func()
def extract_blocks(html: str) -> list[str]:
    tree = HTMLParser(html)
    for n in tree.css("script,style,noscript"):
        n.decompose()

    blocks = []
    for node in tree.css("""title, article, main, p, h1, h2, h3, h4, h5, h6, li, div, section, img[alt], figcaption, caption, blockquote, table th, table td, pre, code, summary, meta[name="description"], meta[property="og:title"], meta[property="og:description"]"""):
        txt = node.text(separator=" ", strip=True)
        if txt: 
            blocks.append(txt)
    return blocks

@daft.func()
def get_block_idx(blocks: list[str]) -> list[int]:
    return list(range(len(blocks)))

def preprocess_common_crawl_html(uri: str, row_limit: int = 1000, index_col: str = "block_id", content_col: str = "block_text"):
    df_warc = daft.read_warc(uri).limit(row_limit)

    df_html = (
        df_warc
        .where(col("WARC-Identified-Payload-Type")== "text/html")
        .with_column("content_raw", remove_http_headers(col("warc_content").try_decode("utf-8")))
        .where(col("content_raw") != "")
    )  

    df_text = (
        df_html
        .with_column("blocks", extract_blocks(col("content_raw")))
        .with_column("block_idx", get_block_idx(col("blocks")))
        .explode("blocks", "block_idx")
        .where(col("blocks") != "")
        .where(col("blocks").not_null())
        .with_column(index_col, col("WARC-Record-ID")+ "-" + col("block_idx"))
        .with_column(content_col, col("blocks"))
        .select(
            "WARC-Record-ID",
            index_col,
            content_col,
        )
    )
    return df_text

def sig(df):  # stable signature
    return set(map(tuple, df.select("u","v").to_pydict().values()))

def diff(a,b):
    A,B = sig(a), sig(b)
    return A-B, B-A

def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(area, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(area, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt

def ee(u: Expression, v: Expression):
    return struct(u.alias("u"), v.alias("v"))

class MinHashDedupePipeline:

    def __init__(self,
        output_uri: str,
        checkpoint_uri: str,
        index_col: str = "block_id", 
        content_col: str = "block_text", 
        component_col: str = "component",
        num_perm: int = 64,
        ngram_size: int = 5,
        threshold: float = 0.7,
        seed: int = 42,
        hash_function: str = 'xxhash',
        remove_punct: bool = True,
        lowercase: bool = False,
        nfd_unicode: bool = True,
        white_space: bool = True,
    ):
        self.output_uri = output_uri
        self.checkpoint_uri = checkpoint_uri
        self.index_col = index_col
        self.content_col = content_col
        self.component_col = component_col
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.seed = seed
        self.hash_function = hash_function
        self.remove_punct = remove_punct
        self.lowercase = lowercase
        self.nfd_unicode = nfd_unicode
        self.white_space = white_space

        B, R = optimal_param(threshold, num_perm)
        assert B * R == num_perm, "B * R must equal num_perm"
        self.B = B
        self.R = R

    def __call__(self, input: DataFrame):
        df_prep = self.prep(input).collect()
        df_norm = self.normalize(df_prep, self.remove_punct, self.lowercase, self.nfd_unicode, self.white_space)
        df_minh = self.minhash(df_norm, self.num_perm, self.ngram_size, self.seed, self.hash_function).collect()
        df_node, id_map = self.prep_node_id_index_map(df_minh)
        df_bands = self.lsh_banding(df_node, self.R, self.B).collect()
        assigns = self.connected_components_2(df_bands)
        assigns_str = self._assignments_back_to_strings(assigns, id_map)
        results = self.merge_results(df_prep, assigns_str)
        return results

        


    # Prepare --------------------------------------------------------------------
    def prep(self, df: DataFrame):
        "Drop Un-needed Columns and add an integer surrogate for index_col"
        return (
            df
            .select(self.index_col, self.content_col)
        )

    # Normalize Text -------------------------------------------------------------
    def normalize(self, 
        df: DataFrame,         
        remove_punct: bool = False,
        lowercase: bool = False,
        nfd_unicode: bool = True,
        white_space: bool = True,
    ) -> DataFrame:
        return (
            df
            .with_column("content_normalized", 
                col(self.content_col).str.normalize(
                    remove_punct=remove_punct, 
                    lowercase=lowercase, 
                    nfd_unicode=nfd_unicode, 
                    white_space=white_space
                )
            )
        )

    # MinHash and Band Generation -------------------------------------------------
    def minhash(self,
        df: DataFrame, 
        num_perm: int = 64, 
        ngram_size: int = 5, 
        seed: int = 42, 
        hash_function: str = 'xxhash',
    ) -> DataFrame:
        # Add monotonically increasing id, and generate minhashes
        return (
            df
            
            .with_column("min_hashes", 
                col("content_normalized").minhash(
                    num_hashes = num_perm,
                    ngram_size = ngram_size,
                    seed = seed, 
                    hash_function = hash_function
                )
            )
            .select(self.index_col, "min_hashes")
        )

    def prep_node_id_index_map(self, df: DataFrame):
        # Add integer index surrogate
        df = df.with_column("node_id", monotonically_increasing_id())
        id_map = df.select(self.index_col, "node_id").distinct()
        return df, id_map

    
    def lsh_banding(self, df: DataFrame, R: int, B: int):
        @daft.func()
        def get_band_idx(band: list[int], B: int) -> list[int]:
            return list(range(min(len(band), B)))

        return (
            df
            .with_column("bands", col("min_hashes").list.chunk(R))
            .with_column("band_idx", get_band_idx(col("bands"), B)) 
            .explode("bands", "band_idx")
            .groupby(col("band_idx"), col("bands"))
            .agg(col("node_id").agg_list().alias("nodes"))
        )
    
    # Connected Components ---------------------------------------------------------
    def _build_edges(self, df: DataFrame):
        return (
            df
            .with_column("u", col("nodes").list.min())
            .explode("nodes")
            .select("u", v=col("nodes"))
            .where(col("u") != col("v"))
            .where(~col("u").is_null())
            .where(~col("v").is_null())
            .distinct()
            .collect()
        )

    def _canonicalize_edges(self, df: DataFrame) -> DataFrame:
        # Always direct from larger id -> smaller id, drop self-loops & dups
        return (
            df
            .select((col("u") <= col("v")).if_else(ee(col("u"), col("v")), ee(col("v"), col("u"))).alias("e"))
            .select(col("e")["*"]) 
            .where(col("u") != col("v"))
            .distinct()
            .collect()
        )

    def _symmetrize(self, df: DataFrame) -> DataFrame:
        # Make undirected by adding both directions
        return (
            df
            .select("u", "v")
            .union_all(df.select(col("v").alias("u"), col("u").alias("v")))
            .collect()
        )

    def _large_star(self, edges: DataFrame) -> DataFrame:
        # Undirected neighborhood via symmetrization
        E = self._symmetrize(edges)

        # Γ(u) as list, m = min(Γ⁺(u))
        neigh = (
            E
            .groupby("u").agg_list("v")
            .with_column("nbrs", col("v"))
            .with_column("m", col("nbrs").list.min())
            .with_column("m", (col("u") < col("m")).if_else(col("u"), col("m")))
        )
        # print("large star neigh:")
        # neigh.show()

        # Emit (v, m(u)) for v > u
        out = (
            neigh.explode("nbrs")
                .where(col("nbrs") > col("u"))
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .where(col("u") != col("v"))
                .distinct()
                .collect()
        )
        # print("large star out where u != v:")
        # out.where(col("u") != col("v")).show()
        return out

    def _small_star(self, edges: DataFrame) -> DataFrame:
        # Direct edges high -> low (canonical)
        directed = self._canonicalize_edges(edges)

        # N(u) = list of lower neighbors v (since u >= v by construction)
        neigh = (
            directed
            .groupby("u").agg_list("v")
            .with_column("nbrs", col("v"))
            .with_column("m", col("nbrs").list.min())
            .with_column("m", (col("u") < col("m")).if_else(col("u"), col("m")))
        )
        # print("small star neigh:")
        # neigh.show()

        # Emit (v, m(u)) for all v in N(u)
        out = (
            neigh.explode("nbrs")
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .where(col("u") != col("v"))
                .distinct()
                .collect()
        )
        # print("small star out where u != v:")
        # out.where(col("u") != col("v")).show()
        return out

    def _check_canonical_set_equality(self, a: DataFrame, b: DataFrame) -> bool:
        ca = self._canonicalize_edges(a)
        cb = self._canonicalize_edges(b)
        # a \ b and b \ a both empty => equal
        left_minus  = ca.join(cb, on=["u","v"], how="anti").count_rows()
        right_minus = cb.join(ca, on=["u","v"], how="anti").count_rows()
        return (left_minus == 0) and (right_minus == 0)

    def _pairs_equal(self, a: DataFrame, b: DataFrame) -> bool:
        """Compare equality of pair sets over columns ["u", "rep"]."""
        left_minus  = a.join(b, on=["u","rep"], how="anti").count_rows()
        right_minus = b.join(a, on=["u","rep"], how="anti").count_rows()
        return (left_minus == 0) and (right_minus == 0)

    def _igraph_connected_components(self, df: DataFrame):
        import igraph as ig
        import pandas as pd
        # Ensure integer dtype and materialize edges
        pdf_edges = (
            df
            .select(col("u").cast(daft.DataType.int64()), col("v").cast(daft.DataType.int64()))
            .where(~col("u").is_null()).where(~col("v").is_null())
            .to_pandas()
        )

        if len(pdf_edges) == 0:
            return set()

        # Build explicit vertex list and index mapping to avoid dtype/label ambiguity
        unique_nodes = pd.unique(pd.concat([pdf_edges["u"], pdf_edges["v"]], ignore_index=True))
        # Convert to Python ints for stable hashing
        node_ids = [int(x) for x in unique_nodes.tolist()]
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Map edges to contiguous indices
        edges_idx = [(id_to_idx[int(u)], id_to_idx[int(v)]) for u, v in zip(pdf_edges["u"], pdf_edges["v"])]
        g = ig.Graph(n=len(node_ids), edges=edges_idx, directed=False)
        comps = g.connected_components(mode="weak")
        # Map back to original node IDs
        return {frozenset(node_ids[i] for i in comp) for comp in comps}

    def _assignments_back_to_strings(self, assigns: DataFrame, id_map: DataFrame) -> DataFrame:
        # assigns: [u(int64), rep(int64)] → [index_col(str), component_col(str)]
        a1 = assigns.join(id_map.with_column_renamed(self.index_col, "__u_str"), left_on="u", right_on="node_id")
        a2 = a1.join(id_map.with_column_renamed(self.index_col, "__rep_str"), left_on="rep", right_on="node_id")
        return a2.select(
            col("__u_str").alias(self.index_col),
            col("__rep_str").alias(self.component_col)
        ).collect()

    def connected_components_2(self, df: DataFrame) -> DataFrame:
        # Start from generated edges; drop nulls and canonicalize
        e = self._build_edges(df)
        b = self._canonicalize_edges(e)

        cc = ConnectedComponents()
        assignments = cc.compute_from_edges(b)

        # Validate assignments vs igraph components derived from the same edge set
        try:
            ig_comps = self._igraph_connected_components(b)
            ours_grouped = (
                assignments
                .groupby("rep")
                .agg(col("u").agg_list().alias("members"))
                .collect()
            )
            pdf = ours_grouped.to_pandas()
            ours_comps = {frozenset(m) for m in pdf["members"]}
            if ours_comps == ig_comps:
                print(f"[VALIDATION] PASSED: components match igraph (n={len(ours_comps)})")
            else:
                only_ours = ours_comps - ig_comps
                only_ig = ig_comps - ours_comps
                def _preview(sets, k=3):
                    out = []
                    for comp in list(sets)[:k]:
                        out.append(sorted(list(comp))[:10])
                    return out
                print(f"[VALIDATION] MISMATCH: ours={len(ours_comps)} vs igraph={len(ig_comps)}")
                print(f"  examples only in ours: {_preview(only_ours)}")
                print(f"  examples only in igraph: {_preview(only_ig)}")
        except Exception as exc:
            print(f"[VALIDATION] Skipped due to error: {exc}")

        return assignments


    def merge_results(self, df: DataFrame, assignment: DataFrame):
        print(f"Original df rows: {df.count_rows()}")
        print(f"Assignment rows: {assignment.count_rows()}")
        
        assignment_unique = (
            assignment
            .select(col(self.index_col), col(self.component_col))
            .groupby(self.index_col)
            .agg(col(self.component_col).min().alias(self.component_col))
        )
        print(f"Assignment unique rows: {assignment_unique.count_rows()}")
        
        df_joined = df.join(assignment_unique, on=self.index_col, how="left")
        print(f"After join rows: {df_joined.count_rows()}")
        
        # Check how many have assignments vs nulls
        assigned_count = df_joined.filter(~col(self.component_col).is_null()).count_rows()
        null_count = df_joined.filter(col(self.component_col).is_null()).count_rows()
        print(f"Assigned: {assigned_count}, Null: {null_count}")
        
        # Check how many are representatives
        rep_count = df_joined.filter(col(self.component_col) == col(self.index_col)).count_rows()
        print(f"Representatives: {rep_count}")
        
        return df_joined.filter(
            col(self.component_col).is_null() | 
            (col(self.component_col) == col(self.index_col))
        ).exclude(self.component_col)
    

def partitioned_save(output_uri: str, df: DataFrame, chunk_size: int, max_partitions: int):
    start_time = time.time()
    total_rows = df.count_rows()
    
    if ray.is_initialized():
        
        partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))
        df = (
            df.repartition(partitions)
            .with_column("__pid__", monotonically_increasing_id() / lit(2**36))
            .write_parquet(output_uri, partition_cols=["__pid__"], write_mode="overwrite", compression="snappy")
        )
    else:
        df = df.write_parquet(output_uri, compression="snappy")

    end_time = time.time()
    print(f"Partitioned Saved {total_rows} rows in {end_time - start_time:.2f}s")
    return df

if __name__ == "__main__":
    # %% Import Libraries, Auth 
    import daft
    from daft.io import IOConfig, S3Config
    import ray
    import pathlib
    from dotenv import load_dotenv

    load_dotenv()

    WORKDIR = pathlib.Path(__file__).parent

    s3_config = S3Config(
        region_name="us-east-1",
        requester_pays=True,
        key_id=os.environ["AWS_ACCESS_KEY_ID"],
        access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        anonymous=False,
    )

    IO_CONFIG = IOConfig(s3=s3_config)
    daft.set_planning_config(default_io_config=IO_CONFIG)

    # %% Define Parameters
    cc_segment = "CC-MAIN-2024-42"
    data_uri = WORKDIR / "data" # For local testing, replace with s3 uri for cloud
    ROW_LIMIT = 50000
    OUTPUT_URI = data_uri / "output"
    CHECKPOINT_URI = data_uri / "checkpoint"

    # MinHash Parameters
    num_perm = 64
    ngram_size = 5
    seed = 42
    hash_function = 'xxhash'
    threshold = 0.7

    # Text Normalization Parameters
    remove_punct = False
    lowercase = False
    nfd_unicode = True
    white_space = True

    # Partitioned Save Parameters
    chunk_size = 200_000
    max_partitions = 2048
    persist_checkpoint = True

    # %% Initialize Dedupe Pipeline
    pipeline = MinHashDedupePipeline(
        output_uri=OUTPUT_URI,
        checkpoint_uri=CHECKPOINT_URI,
        index_col="block_id",
        content_col="block_text",
        component_col="component",
        num_perm = num_perm,
        ngram_size = ngram_size,
        threshold = threshold,
        seed = seed,
        hash_function = hash_function,
        remove_punct = remove_punct,
        lowercase = lowercase,
        nfd_unicode = nfd_unicode,
        white_space = white_space,
    )


    # PIPELINE -------------------------------------------------------------
    
    # %% Preprocess Common Crawl HTML and Persist
    start_time = time.time()
    df_prepped = preprocess_common_crawl_html(
        uri=f"s3://commoncrawl/crawl-data/{cc_segment}/segments/*/warc/*.warc.gz",
        row_limit=ROW_LIMIT,
        index_col="block_id",
        content_col="block_text",
    ).collect()
    #partitioned_save(CHECKPOINT_URI / "prepped", df_prepped, chunk_size, max_partitions)
    # %% Run Pipeline
    #df_prepped = daft.read_parquet(str(CHECKPOINT_URI / "prepped")).limit(10000)
    df_results = pipeline(df_prepped).collect()
    
    # Stage 5: Print Results
    prepped_rows = df_prepped.count_rows()
    results_rows = df_results.count_rows()
    print("─" * 80)
    print(f"# of rows before:  {prepped_rows}")
    print(f"# of rows after:   {results_rows}")
    print(f"% of rows kept:    {results_rows / prepped_rows * 100:.2f}%")
    print(f"Output Directory:  {OUTPUT_URI}")
    print(f"Overall Time:      {time.time() - start_time:.2f}s")
    print("─" * 80)

    partitioned_save(OUTPUT_URI, df_results, chunk_size, max_partitions)

