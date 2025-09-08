
import os
from selectolax.parser import HTMLParser
import math
import time

import daft
from daft import col, lit, struct, DataFrame, Expression
from daft.functions import monotonically_increasing_id
from daft.io import IOConfig, S3Config

from scipy.integrate import quad as integrate
from typing import Literal


from logging import getLogger, INFO

logger = getLogger(__name__)
logger.setLevel(INFO)


@daft.func()
def remove_http_headers(x: str) -> str:
    """Remove HTTP headers from input string by splitting on double CRLF, returning the body or empty string."""
    if x is None:
        return ""
    if len(x.split("\r\n\r\n")) > 1:
        return x.split("\r\n\r\n")[1]
    return ""

@daft.func()
def extract_blocks(html: str) -> list[str]:
    """Parse HTML using Selectolax, remove scripts/styles/noscripts, extract text blocks from key tags."""
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
    """Generate integer indices for each element in the input list of blocks."""
    return list(range(len(blocks)))

def preprocess_common_crawl_html(uri: str, row_limit: int = 1000, index_col: str = "block_id", content_col: str = "block_text"):  # pragma: no cover
    """Preprocess Common Crawl WARC records: filter HTML, remove headers, extract text blocks, add unique IDs. Returns daft DataFrame with index and content columns."""
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



def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """Compute optimal LSH bands (b) and rows (r) minimizing weighted FP/FN for given threshold and permutations. Returns (b, r) tuple."""

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
    """Create a struct Expression with fields 'u' and 'v' for representing edges."""
    return struct(u.alias("u"), v.alias("v"))

def sig(df):  # stable signature
    """Convert DataFrame of u,v columns to a set of (u,v) tuples for stable signature comparison."""
    return set(map(tuple, df.select("u","v").to_pydict().values()))

def diff(a,b):
    """Compute set differences between signatures of two DataFrames: returns (A-B, B-A)."""
    A,B = sig(a), sig(b)
    return A-B, B-A
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
        igraph_validate: bool = False,
        algorithm: Literal["alternating", "two_phase"] = "alternating",
        max_loops: int = 100,
    ):
        """Initialize the MinHash deduplication pipeline with configuration for text processing, LSH, and connected components."""
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
        self.igraph_validate = igraph_validate
        self.algorithm = algorithm
        self.max_loops = max_loops

        B, R = optimal_param(threshold, num_perm)
        assert B * R == num_perm, "B * R must equal num_perm"
        self.B = B
        self.R = R

    def __call__(self, input: DataFrame):
        """Execute the full deduplication pipeline on the input DataFrame, returning deduplicated results."""
        df_prep = self.prep(input).collect()
        df_norm = self.normalize(df_prep, self.remove_punct, self.lowercase, self.nfd_unicode, self.white_space)
        df_minh = self.minhash(df_norm, self.num_perm, self.ngram_size, self.seed, self.hash_function)
        df_node, id_map = self.prep_node_id_index_map(df_minh)
        df_bands = self.lsh_banding(df_node, self.R, self.B)
        assigns = self.connected_components_2(df_bands, self.algorithm, self.max_loops, self.igraph_validate)
        results = self.merge_results(df_prep, assigns, id_map)
        return results

    
    # Prepare --------------------------------------------------------------------
    def prep(self, df: DataFrame):
        """Prepare input DataFrame by selecting index and content columns for downstream processing."""
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
        """Apply text normalization to content column with optional flags for punctuation, case, Unicode, and whitespace."""
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
        """Compute MinHash signatures on normalized content using specified parameters."""
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
        """Add integer node IDs to DataFrame and create mapping back to original string indices."""
        # Add integer index surrogate
        df = df.with_column("node_id", monotonically_increasing_id())
        id_map = df.select(self.index_col, "node_id").distinct()
        return df, id_map

    
    def lsh_banding(self, df: DataFrame, R: int, B: int):
        """Band MinHash signatures and group nodes by band index and hash for candidate pairs."""
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
        """Build edge list from grouped nodes by connecting each to the minimum in group."""
        return (
            df
            .with_column("u", col("nodes").list.min())
            .explode("nodes")
            .select("u", v=col("nodes"))
            .where(col("u") != col("v"))
            .where(~col("u").is_null())
            .where(~col("v").is_null())
            .distinct()
            
        )
    
    def _large_star(self, edges: DataFrame) -> DataFrame:
        """Perform large-star operation: connect nodes to min in extended neighborhood."""
        # 1. Emit U,V and V,U 
        undirected = (
            edges
            .select("u", "v")
            .union_all(edges.select(col("v").alias("u"), col("u").alias("v")))
            
        )

        # Step 2: Group by u, and aggregate the list of v's
        neigh = (
            undirected
            .groupby("u").agg_list("v")
            .with_column("nbrs", col("v"))
        )

        # Step 3: Compute m = min over nbrs union {u}
        neigh = neigh.with_column("m", col("nbrs").list.min())
        neigh = neigh.with_column(
            "m", 
            col("m").is_null().if_else(
                col("u"),
                (col("u") < col("m")).if_else(col("u"), col("m"))
            )
        )

        # Step 4: Emit (v, m(u)) for v > u
        out = (
            neigh.explode("nbrs")
                .where(col("nbrs") > col("u")) 
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .where(col("u") != col("v"))
                .distinct()
                
        )

        return out

    def _small_star(self, edges: DataFrame) -> DataFrame:
        """Perform small-star operation: connect to min in direct smaller neighborhood."""
        # Step 1: For each edge, emit to the larger node as key, smaller as value
        directed =  (
            edges.select(
                (col("u") < col("v")).if_else(
                    ee(col("u"), col("v")), 
                    ee(col("v"), col("u"))
                ).alias("e"))
            .select(col("e")["*"]) 
            .where(col("u") != col("v"))
            .distinct()
        )

        # Step 2: Group by larger u, nbrs are smaller neighbors
        neigh = (
            directed
            .groupby("u").agg_list("v")
            .with_column("nbrs", col("v"))
        )

        # Step 3: Compute m = min over nbrs union {u}
        neigh = neigh.with_column("m", col("nbrs").list.min())
        neigh = neigh.with_column(
            "m", 
            col("m").is_null().if_else(
                col("u"),
                (col("u") < col("m")).if_else(col("u"), col("m"))
            )
        )

        # Emit (v, m(u)) for all v in N(u)
        out = (
            neigh.explode("nbrs")
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .where(col("u") != col("v"))
                .distinct()
                
        )
        
        return out

    def canonicalize(self, edges: DataFrame) -> DataFrame:
        """Order edges so u < v and deduplicate for canonical representation."""
        return (
            edges
            .with_column("u_can", (col("u") < col("v")).if_else(col("u"), col("v")))
            .with_column("v_can", (col("u") < col("v")).if_else(col("v"), col("u")))
            .select(col("u_can").alias("u"), col("v_can").alias("v"))
            .distinct()
        )

    def symmetrize(self, edges: DataFrame) -> DataFrame:
        """Make edge list undirected by adding reverse edges."""
        return (
            edges
            .select("u", "v")
            .union_all(edges.select(col("v").alias("u"), col("u").alias("v")))
            
        )

    def check_canonical_set_equality(self, prev_edges: DataFrame, curr_edges: DataFrame) -> bool:
        """Check if two edge DataFrames represent the same set after canonicalization."""
        prev_can = self.canonicalize(prev_edges).to_pydict()
        curr_can = self.canonicalize(curr_edges).to_pydict()
        prev_set = set(zip(prev_can["u"], prev_can["v"]))
        curr_set = set(zip(curr_can["u"], curr_can["v"]))
        return prev_set == curr_set

    
    def construct_assignments(self, b: DataFrame) -> DataFrame:
        """Build node-to-representative assignments from edge list, using min neighbor."""
        # Build the set of all unique node IDs that appear in the edge list
        # (both as source 'u' and destination 'v')
        nodes = (
            b.select(col("u").alias("u"))          # grab all source nodes
             .union_all(b.select(col("v").alias("u")))  # grab all destination nodes
             .distinct()                           # deduplicate to get unique nodes
        )
        
        # For every node, compute the smallest node ID it is connected to
        # (i.e., its tentative representative / root in the current component)
        rep_map = (
            b
            .groupby("u")                          # group edges by source node
            .agg(col("v").min().alias("rep"))      # find the smallest neighbor
        )
        
        # Join each node with its tentative representative.
        # Nodes that have no outgoing edges (and thus no entry in rep_map)
        # become their own representative.
        assignments = (
            nodes
            .join(rep_map, on="u", how="left")     # left join to keep all nodes
            .with_column(
                "rep",
                col("rep").is_null()               # if no neighbor was found
                .if_else(col("u"), col("rep"))     # use the node itself as rep
            )
            .select("u", "rep")                    # keep only node and its rep
            .distinct()                            # deduplicate any duplicates
                                         # materialize the result
        )
        return assignments

    def _pairs_equal(self, a: DataFrame, b: DataFrame) -> bool:
        """Check if two DataFrames have identical (u, rep) pairs via anti-joins."""
        left_minus  = a.join(b, on=["u","rep"], how="anti").count_rows()
        right_minus = b.join(a, on=["u","rep"], how="anti").count_rows()
        return (left_minus == 0) and (right_minus == 0)

    def global_min_label_propagation(self, b: DataFrame, assignments: DataFrame) -> DataFrame:
        """Propagate global minimum labels across components until convergence for consistency."""
        """
        Propagate the global minimum label across the undirected graph until convergence.

        Why this is needed:
        - After alternating Large-/Small-Star and applying path compression, components can still
          stabilize with multiple local minima (distinct labels) within the same true component.
        - This deterministic min-label diffusion ensures every node in a connected component adopts
          the single global minimum node-id as its representative, restoring exact parity to igraph.

        Inputs:
        - b: DataFrame of edges with columns ["u", "v"]. Direction does not matter here; we symmetrize.
        - assignments: DataFrame of current labels with columns ["u", "rep"].

        Algorithm:
        1) Symmetrize edges to build an undirected adjacency (both directions present).
        2) Initialize labels(u) from assignments.rep.
        3) Iterate up to lp_max_iters times:
           a) For each node, compute nbr_min(u) = min(label(v)) over neighbors v of u.
           b) Update label(u) = min(label(u), nbr_min(u)) with null-safe handling.
           c) Deduplicate and compare to prior labels; stop when the (u, label) pair set stabilizes.
        4) Return labels as assignments with schema ["u", "rep"].

        Complexity:
        - Each iteration is a small number of scans and groupbys; convergence is fast in practice.
        Determinism:
        - Pure min-reduction yields a unique fixed point given the input edges and initial labels.
        """
        # Build an undirected view of the graph so labels can flow in both directions
        E = self.symmetrize(b)

        # Initialize labels from current assignments: rep becomes the working label per node
        labels = assignments.select(col("u"), col("rep").alias("label"))

        lp_iters = 0
        lp_max_iters = 100
        while lp_iters < lp_max_iters:
            lp_iters += 1

            # For each node u, compute the minimum label among its neighbors
            nbr_min = (
                E
                .join(labels, left_on="v", right_on="u", how="left")
                .select(col("u").alias("node"), col("label"))
                .groupby("node")
                .agg(col("label").min().alias("nbr_min"))
                
            )

            # Lower each node's label to min(current_label, neighbor_min_label)
            labels_next = (
                labels
                .join(nbr_min, left_on="u", right_on="node", how="left")
                .with_column(
                    "label",
                    col("nbr_min").is_null().if_else(
                        col("label"),
                        (col("label") <= col("nbr_min")).if_else(col("label"), col("nbr_min")),
                    ),
                )
                .select(col("u"), col("label"))
                .distinct()
            )

            # Convergence: compare pair sets after casting back to (u, rep)
            if self._pairs_equal(
                assignments.select(col("u"), col("rep").alias("label")).select(col("u"), col("label").alias("rep")).collect(),
                labels_next.select(col("u"), col("label").alias("rep")).collect(),
            ):
                break

            # Continue iterating with updated assignments/labels
            assignments = labels_next.select(col("u"), col("label").alias("rep")).collect()
            labels = labels_next

        return assignments

    def connected_components_2(self,
        df: DataFrame,
        algorithm: Literal["alternating", "two_phase"] = "alternating",
        max_loops: int = 100,
        igraph_validate: bool = False,
    ) -> DataFrame:
        """Compute connected components using star operations, propagation, and optional igraph validation."""
        # Start from generated edges; drop nulls and canonicalize
        e = self._build_edges(df)
        
        if igraph_validate:
            ig_comps = self._igraph_connected_components(e)

        b = e
        if algorithm == "alternating":
            for _ in range(max_loops):
                a = self._large_star(b)
                b_next = self._small_star(a)

                if self.check_canonical_set_equality(b, b_next):
                    b = b_next
                    break
                b = b_next

        elif algorithm == "two_phase":
            # Outer Loop - Repeat (large-star to fixed point) THEN one small-star
            for _ in range(max_loops):
                L = b
                # Inner Loop: large-star until no change
                for _ in range(max_loops):
                    L_next = self._large_star(L)
                    if self.check_canonical_set_equality(L, L_next):
                        L = L_next
                        break
                    L = L_next

                b_next = self._small_star(L)
                if self.check_canonical_set_equality(b, b_next):
                    b = b_next
                    break
                b = b_next

        assignments = self.construct_assignments(b)
        
        # Ensures igraph parity 
        assignments = self.global_min_label_propagation(b, assignments)
        
        if igraph_validate:
            self._igraph_validate_assignments(assignments, ig_comps)

        return assignments

    def _igraph_connected_components(self, df: DataFrame):  # pragma: no cover
        """Compute connected components using igraph library for validation purposes."""
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


    def _igraph_validate_assignments(self, assignments: DataFrame, ig_comps: set):  # pragma: no cover
        """Validate computed assignments against igraph components, logging mismatches."""
        # Validate assignments vs igraph components derived from the same edge set
        try:
            ours_grouped = (
                assignments
                .groupby("rep")
                .agg(col("u").agg_list().alias("members"))
                
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
                        out.append(sorted(list(int(c) for c in comp))[:10])
                    return out
                print(f"[VALIDATION] MISMATCH: ours={len(ours_comps)} vs igraph={len(ig_comps)}")
                print(f"  examples only in ours: {_preview(only_ours)}")
                print(f"  examples only in igraph: {_preview(only_ig)}")
        except Exception as exc:
            print(f"[VALIDATION] Skipped due to error: {exc}")

    def _assignments_back_to_strings(self, assigns: DataFrame, id_map: DataFrame) -> DataFrame:
        """Map integer node assignments back to original string index values."""
        # assigns: [u(int64), rep(int64)] → [index_col(str), component_col(str)]
        a1 = assigns.join(id_map.with_column_renamed(self.index_col, "__u_str"), left_on="u", right_on="node_id")
        a2 = a1.join(id_map.with_column_renamed(self.index_col, "__rep_str"), left_on="rep", right_on="node_id")
        return a2.select(
            col("__u_str").alias(self.index_col),
            col("__rep_str").alias(self.component_col)
        )

    def merge_results(self, df: DataFrame, assignment: DataFrame, id_map: DataFrame):
        """Merge assignments with original DataFrame, filtering to unique representatives per component."""
        # Select minimum integer representative per component
        assignment_unique = (
            assignment
            .groupby("u")
            .agg(col("rep").min())
        )
        # Map integer representatives back to original string indices
        assignments_unique_str = self._assignments_back_to_strings(assignment_unique, id_map)

        # Join back to original df and filter to keep only rows where the row is its own representative or isolated
        df_joined = df.join(assignments_unique_str, on=self.index_col, how="left")
        
        return (
            df_joined
            .filter(
                col(self.component_col).is_null() | 
                (col(self.component_col) == col(self.index_col))
            )
            .exclude(self.component_col)
        )
    

def partitioned_save(output_uri: str, df: DataFrame, chunk_size: int, max_partitions: int):
    """Save DataFrame to partitioned Parquet, using Ray for repartitioning if available."""
    start_time = time.time()
    total_rows = df.count_rows()
    
    # Import ray lazily so environments without ray can still use the fallback path
    try:
        import ray  # type: ignore
    except Exception:
        ray = None  # type: ignore

    if (ray is not None) and getattr(ray, "is_initialized", lambda: False)():
        
        partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))
        df = (
            df.repartition(partitions)
            .with_column("__pid__", monotonically_increasing_id() / lit(2**36))
            .write_parquet(output_uri, partition_cols=["__pid__"], write_mode="overwrite", compression="snappy")
        )
    else:
        df.write_parquet(output_uri, compression="snappy")

    end_time = time.time()
    print(f"Partitioned Saved {total_rows} rows in {end_time - start_time:.2f}s")
    return df

if __name__ == "__main__":  # pragma: no cover
    # %% Import Libraries, Auth 
    import daft
    from daft.io import IOConfig, S3Config
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
    ROW_LIMIT = 100
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
        igraph_validate = True,
        algorithm = "two_phase",
        max_loops = 100,
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

