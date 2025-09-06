
import os
from selectolax.parser import HTMLParser
import re
import hashlib
import math
import time

import daft
from daft import col, lit, struct, DataFrame, Expression
from daft.functions import monotonically_increasing_id
from daft.io import IOConfig, S3Config

from scipy.integrate import quad as integrate


from logging import getLogger, INFO

from .utils import optimal_param, checkpoint, partitioned_save, log_results

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


def ee(u: Expression, v: Expression):
    return struct(u.alias("u"), v.alias("v")).alias("e")

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

    def run(self,
        cc_warc_uri: str,
        row_limit: int = 1000,
        checkpoints_active = True,

    ):
        
        start_time = time.time()
        

        # Stage 1: Preprocess HTML to extract text
        df_raw = self.load_data(cc_warc_uri, row_limit)
        df_prepped = self.preprocess(df_raw)

        df_prepped = checkpoint(self.checkpoint_uri, df_prepped, "prepped", checkpoints_active)

        # Stage 2: Normalize Text, MinHash, and Band Generation
        df_norm    = self.normalize(df_prepped, self.remove_punct, self.lowercase, self.nfd_unicode, self.white_space)
        df_minhash = self.minhash(df_norm, self.num_perm, self.ngram_size, self.seed, self.hash_function)
        df_band    = self.band_generation(df_minhash, self.R, self.B)
        df_grouped = self.group_bands(df_band)

        df_grouped = checkpoint(self.checkpoint_uri, df_grouped, "bands", checkpoints_active)
        
        # Stage 3: Connected Components 
        df_edges = self._generate_edges(df_grouped)
        df_edges = checkpoint(self.checkpoint_uri, df_edges, "edges", checkpoints_active)


        df_assignments2 = self.connected_components_2(df_edges)
        df_assignments2 = checkpoint(self.checkpoint_uri, df_assignments2, "assignments2", checkpoints_active)
        df_assignments2.show()





        df_assignments = self.connected_components(df_edges)
        df_assignments = checkpoint(self.checkpoint_uri, df_assignments, "assignments", checkpoints_active)
        df_assignments.show()





        # Stage 4: Merge Results
        df_results = self.merge_results(df_prepped, df_assignments)
        df_results = checkpoint(self.checkpoint_uri, df_results, "results", checkpoints_active)

        # Stage 5: Write Results
        end_time = time.time()
        self.log_results(df_prepped, df_results, start_time, end_time)
        self.partitioned_save(df_results, chunk_size, max_partitions)

        return df_results    

    # Load Data ------------------------------------------------------------------
    def load_data(self, uri: str, row_limit: int = 1000):
        return daft.read_warc(uri).limit(row_limit)

    # Preprocess HTML to extract text ----------------------------------------------
    def preprocess(self, df: DataFrame):
        "Filters payloads to just text/html, removes http headers, and adds a monotonically increasing id"

        return (
            df
            .where(col("WARC-Identified-Payload-Type")== "text/html")
            .with_column("content_raw", remove_http_headers(col("warc_content").try_decode("utf-8")))
            .where(col("content_raw") != "")
            .with_column("content_text", extract_blocks(col("content_raw")).list.join(" "))
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
                col("content_text").str.normalize(
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
    
    def band_generation(self, df: DataFrame, R: int, B: int):
        return (
            df
            .with_column("bands", col("min_hashes").list.chunk(R))
            .with_column("band_idx", lit(list(range(B))))
            .explode("bands", "band_idx")
        )
    
    def group_bands(self, df: DataFrame):
        return (
            df
            .groupby(col("band_idx"), col("bands"))
            .agg(col(self.index_col).agg_list().alias("nodes"))
        )
    
    # Connected Components ---------------------------------------------------------
    def _generate_edges(self, df: DataFrame):
        return (
            df
            .with_column("left_edge", col("nodes").list.min())
            .explode("nodes")
            .select("left_edge", right_edge=col("nodes"))
            .filter(col("left_edge") != col("right_edge"))
            .distinct()
        )

    def _build_node_id_map(self, edges: DataFrame) -> DataFrame:
        # Gather all node IDs (as strings), assign unique int64 IDs
        nodes = (
            edges.select(col("left_edge").alias(self.index_col))
                .union_all(edges.select(col("right_edge").alias(self.index_col)))
                .where(~col(self.index_col).is_null())
                .distinct()
                .with_column("node_id", monotonically_increasing_id().cast(daft.DataType.int64()))
                .collect()
        )
        return nodes  # [self.index_col (str), node_id (int64)]

    def _edges_to_numeric(self, edges: DataFrame, id_map: DataFrame) -> DataFrame:
        # Join twice to get numeric u,v
        left = edges.join(id_map, left_on="left_edge", right_on=self.index_col).rename({ "node_id": "u" })
        both = left.join(id_map, left_on="right_edge", right_on=self.index_col).rename({ "node_id": "v" })
        return both.select(col("u").cast(daft.DataType.int64()), col("v").cast(daft.DataType.int64())).collect()

    def _assignments_back_to_strings(self, assigns: DataFrame, id_map: DataFrame) -> DataFrame:
        # assigns: [u(int64), rep(int64)] → [index_col(str), component_col(str)]
        a1 = assigns.join(id_map.rename({self.index_col: "__u_str"}), left_on="u", right_on="node_id")
        a2 = a1.join(id_map.rename({self.index_col: "__rep_str"}), left_on="rep", right_on="node_id")
        return a2.select(
            col("__u_str").alias(self.index_col),
            col("__rep_str").alias(self.component_col)
        ).collect()

    def _canonicalize_edges(self, df: DataFrame) -> DataFrame:
        # Always direct from larger id -> smaller id, drop self-loops & dups
        return (
            df.select(
                (col("u") >= col("v")).if_else(col("u"), col("v")).alias("u"),
                (col("u") >= col("v")).if_else(col("v"), col("u")).alias("v"),
            )
            .where(col("u") != col("v"))
            .distinct()
            .collect()
        )

    def _symmetrize(self, df: DataFrame) -> DataFrame:
        # Make undirected by adding both directions
        return (
            df.select("u", "v")
            .union_all(df.select(col("v").alias("u"), col("u").alias("v")))
            .distinct()
            .collect()
        )

    def _large_star_phase_2(self, edges: DataFrame) -> DataFrame:
        # Undirected neighborhood via symmetrization
        E = self._symmetrize(edges)

        # Γ(u) as list, m = min(Γ⁺(u))
        neigh = (
            E.groupby("u")
            .agg(col("v").agg_list().alias("nbrs"))
            .with_column("m", col("nbrs").list.min())
            .with_column("m", (col("u") < col("m")).if_else(col("u"), col("m")))
        )

        # Emit (v, m(u)) for v > u
        out = (
            neigh.explode("nbrs")
                .where(col("nbrs") > col("u"))
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .distinct()
                .collect()
        )
        return out

    def _small_star_phase_2(self, edges: DataFrame) -> DataFrame:
        # Direct edges high -> low (canonical)
        directed = self._canonicalize_edges(edges)

        # N(u) = list of lower neighbors v (since u >= v by construction)
        neigh = (
            directed.groupby("u")
                    .agg(col("v").agg_list().alias("nbrs"))
                    .with_column("m", col("nbrs").list.min())
                    .with_column("m", (col("u") < col("m")).if_else(col("u"), col("m")))
        )

        # Emit (v, m(u)) for all v in N(u)
        out = (
            neigh.explode("nbrs")
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .distinct()
                .collect()
        )
        return out

    def _check_convergence_2(self, a: DataFrame, b: DataFrame) -> bool:
        ca = self._canonicalize_edges(a)
        cb = self._canonicalize_edges(b)
        # a \ b and b \ a both empty => equal
        left_minus  = ca.join(cb, on=["u","v"], how="anti").count_rows()
        right_minus = cb.join(ca, on=["u","v"], how="anti").count_rows()
        return (left_minus == 0) and (right_minus == 0)

    def connected_components_2(self, df: DataFrame) -> DataFrame:
        # Start from generated edges; drop nulls and canonicalize
        b = (
            df.select(col("left_edge").alias("u"), col("right_edge").alias("v"))
            .where(~col("u").is_null()).where(~col("v").is_null())
            .collect()
        )
        b = self._canonicalize_edges(b)

        # Optional: sanity check with igraph
        print(self.igraph_connected_components(b))

        # Star contraction
        while True:
            a = self._large_star_phase_2(b)
            b = self._small_star_phase_2(a)
            if self._check_convergence_2(a, b):
                break

        # After convergence, choose minimal representative per node
        assignments = (
            b.groupby("u").agg(col("v").min().alias("rep"))
            .select(col("u").alias(self.index_col), col("rep").alias(self.component_col))
            .collect()
        )
        assignments.show()
        return assignments



    def _large_star_phase(self, df: DataFrame):
        """ Large-Star Operation 
        
        1: MAP <u; v>:
        2:     Emit <u; v> and <v; u>.
        3: Reduce <u; Γ(u)>:
        4:     Let m = argmin_(v ∈ Γ+(u)) of l_v
        5:     Emit <v; m> for all v where l_v > l_u.

        See: https://dl.acm.org/doi/pdf/10.1145/2670979.2670997 (PDF)

        Citation:
        Raimondas Kiveris, Silvio Lattanzi, Vahab Mirrokni, Vibhor Rastogi, and Sergei Vassilvitskii. 
        2014. 
        Connected Components in MapReduce and Beyond. In Proceedings of the ACM Symposium on Cloud Computing (SOCC '14). 
        Association for Computing Machinery, New York, NY, USA, 1–13. 
        https://doi.org/10.1145/2670979.2670997
        """
        return (df
            # large_star_map
            .select("u", "v")
            .union_all(df.select(col("v").alias("u"), col("u").alias("v")))
            
            # large_star_reduce
            .groupby("u").agg_list("v")
            .with_column("min_edge", col("v").list.min()) # Get minimum of v neighbors
            .with_column("min_edge", (col("u") <= col("min_edge")).if_else(col("u"), col("min_edge"))) # Get minimum of u and min_edge
            .explode("v")
            .where(col("v") > col("u"))   
            .with_column("e", ee(col("v"), col("min_edge")))

            # Label and remove nulls, duplicates
            .select("e")
            .where(~col("e").is_null())
            .distinct()
            .select(col("e")["*"])
            .where(col("u") != col("v"))
            .collect()
        )


    
    def _small_star_phase(self, df: DataFrame):
        """ Small-Star Operation 
        
        1: MAP <u; v>:
        2: if l_v ≤ l_u then: # (if u >= v)
        3:     Emit <u; v>
        4: else
        5:     Emit <v; u>

        7: Reduce <u; N ⊆ Γ(u)>:
        8:     Let m = argmin v ∈ N ∪ {u} `v.
        9:     Emit <v;m> for all v ∈ N.

        See: https://dl.acm.org/doi/pdf/10.1145/2670979.2670997 (PDF)

        Citation:
        Raimondas Kiveris, Silvio Lattanzi, Vahab Mirrokni, Vibhor Rastogi, and Sergei Vassilvitskii. 
        2014. 
        Connected Components in MapReduce and Beyond. In Proceedings of the ACM Symposium on Cloud Computing (SOCC '14). 
        Association for Computing Machinery, New York, NY, USA, 1–13. 
        https://doi.org/10.1145/2670979.2670997
        """
        return (
            df
            # small_star_map (emit)
            .select((col("u") >= col("v")).if_else(ee(col("u"), col("v")), ee(col("v"), col("u"))).alias("e"))
            .select(col("e")["*"]) 
            
            # small_star_reduce
            .groupby("u").agg_list("v")
            .with_column("min_edge", col("v").list.min())
            .with_column("min_edge", (col("u") < col("min_edge")).if_else(col("u"), col("min_edge")))
            .explode("v")  
            .with_column("e", ee(col("v"), col("min_edge")))

            # Clean up label, remove nulls/duplicates
            .select("e")
            .where(~col("e").is_null())
            .distinct()
            .select(col("e")["*"])

            .collect()
        )
    
    def _check_convergence(self, a: DataFrame, b: DataFrame):
        a_hash = a.select(col("u").hash().alias("hash")).sum("hash").to_pydict()["hash"][0]
        b_hash = b.select(col("u").hash().alias("hash")).sum("hash").to_pydict()["hash"][0]
        if a_hash == b_hash:
            return True
        return False
        
    
    def igraph_connected_components(self, df: DataFrame):
        import igraph as ig
        df = df.select(col("u").cast(daft.DataType.int64()), col("v").cast(daft.DataType.int64())).to_pandas()
        g = ig.Graph.DataFrame(df, directed=False)
        return {frozenset(c) for c in g.connected_components(mode="weak") if len(c) > 1}
    

    def connected_components(self, df: DataFrame):
        # Initialize b
        b = (
            df.select(col("left_edge").alias("u"), col("right_edge").alias("v"))
            .where(~col("u").is_null())
            .where(~col("v").is_null())
            .collect() # Materialize
        )    

        #b = self._canonicalize_edges(b)
        ig_components = self.igraph_connected_components(b)
        print(ig_components)

        # Star Contraction
        while True:
            a = self._large_star_phase(b)
            a.show()
            b = self._small_star_phase(a)
            b.show()
            
            if self._check_convergence(a, b):
                break
        
        # Return contracted star edges
        return (
            b
            .select(col("u").alias(self.index_col), col("v").alias(self.component_col))
            .collect() # Materialize
        )

    def merge_results(self, df: DataFrame, assignment: DataFrame):
        df = df.into_batches(100_000)
        df = df.join(
            assignment.select(col(self.index_col), col(self.component_col)),
            on=self.index_col,
            how="left",
        )
        df = (
            df
            .filter(col(self.component_col).is_null() | (col(self.component_col) == col(self.index_col)))
            .exclude(self.component_col)
        )

        return df
    





if __name__ == "__main__":
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

    # Define Parameters
    cc_segment = "CC-MAIN-2024-42"
    save_uri = WORKDIR / "data" # For local testing, replace with s3 uri for cloud
    ROW_LIMIT = 500
    
    

    # MinHash Parameters
    num_perm = 64
    ngram_size = 5
    seed = 42
    hash_function = 'xxhash'
    threshold = 0.717

    # Text Normalization Parameters
    remove_punct = False
    lowercase = False
    nfd_unicode = True
    white_space = True

    # Partitioned Save Parameters
    chunk_size = 200_000
    max_partitions = 2048
    persist_checkpoint = True


    # PIPELINE -------------------------------------------------------------
    # Preprocess Common Crawl HTML and Persist
    df_prepped = preprocess_common_crawl_html(
        uri=f"s3://commoncrawl/crawl-data/{cc_segment}/segments/*/warc/*.warc.gz",
        row_limit=ROW_LIMIT,
        index_col="block_id",
        content_col="block_text",
        component_col="component",
    )

    if ray.is_initialized():
        partitioned_save(df_prepped, save_uri, chunk_size, max_partitions)
    else:
        df_prepped = df_prepped.write_parquet(save_uri+ "/prepped", compression="snappy")

    # Run it
    mh_pipeline= MinHashDedupePipeline(
        output_uri=save_uri + "/output",
        checkpoint_uri=save_uri + "/checkpoint",
        index_col="block_id",
        content_col="block_text",
        component_col="component",
    )
    results = mh_pipeline.run(
        prepped_uri=save_uri + "/prepped",

        checkpoints_active=persist_checkpoint,
        remove_punct=remove_punct,
        lowercase=lowercase,
        nfd_unicode=nfd_unicode,   
        white_space=white_space,
        num_perm=num_perm,
        ngram_size=ngram_size,   
        seed=seed,
        hash_function=hash_function,
        threshold=threshold,
    )