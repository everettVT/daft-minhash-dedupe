
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
    for n in tree.css("script,style,noscript"):  n.decompose()
    blocks = []
    for node in tree.css("article, main, p, h1, h2, h3, li"):
        txt = node.text(separator=" ", strip=True)
        if not txt: 
            continue
        blocks.append(txt)
    return blocks


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
    return struct(u.alias("u"), v.alias("v")).alias("e")

@daft.func(return_dtype=daft.DataType.list(daft.DataType.uint64()))
def combine(u: int, v: list[int]):
    return [u, *v]


class CommonCrawlHtmlMinHashDedupe:

    def __init__(self, 
        output_uri: str,
        checkpoint_uri: str,
        index_col: str = "__id__", 
        content_col: str = "__content_text__", 
        component_col: str = "__component__",
    ):
        
        self.output_uri = output_uri
        self.checkpoint_uri = checkpoint_uri
        self.index_col = index_col
        self.content_col = content_col
        self.component_col = component_col


    def __call__(self,
        cc_warc_uri: str,
        row_limit: int = 1000,
        chunk_size: int = 200_000,
        max_partitions: int = 2048,
        persist_checkpoint: bool = True,
        remove_punct: bool = False,
        lowercase: bool = False,
        nfd_unicode: bool = True,
        white_space: bool = True,
        num_perm: int = 64,
        ngram_size: int = 5,
        seed: int = 42,
        hash_function: str = 'xxhash',
        threshold: float = 0.717,

    ):
        start_time = time.time()
        # Stage 1: Preprocess HTML to extract text
        df_raw = self.load_data(cc_warc_uri, row_limit)
        df_prepped = self.preprocess(df_raw)
        df_prepped = self.checkpoint(df_prepped, "prepped", persist_checkpoint)

        # Stage 2: Normalize Text, MinHash, and Band Generation
        df_norm = self.normalize(df_prepped, remove_punct, lowercase, nfd_unicode, white_space)
        df_minhash = self.minhash(df_norm, num_perm, ngram_size, seed, hash_function)
        B, R = optimal_param(threshold, num_perm)
        df_band = self.band_generation(df_minhash, R, B)
        df_grouped = self.group_bands(df_band)
        df_grouped = self.checkpoint(df_grouped, "bands", persist_checkpoint)
        
        # Stage 3: Connected Components
        df_edges = self._generate_edges(df_grouped)
        df_assignments = self.connected_components(df_edges)
        df_assignments = self.checkpoint(df_assignments, "assignments", persist_checkpoint)
        df_assignments.show()
        # Stage 4: Merge Results
        df_results = self.merge_results(df_prepped, df_assignments)
        df_results = self.checkpoint(df_results, "results", persist_checkpoint)

        # Stage 5: Write Results
        end_time = time.time()
        self.log_results(df_prepped, df_results, start_time, end_time)
        self.partitioned_save(df_results, chunk_size, max_partitions)

        return df_results

    # Checkpoint ------------------------------------------------------------------
    def checkpoint(self, df: daft.DataFrame, stage: str, persist_checkpoint: bool = True):
        uri = f"{self.checkpoint_uri}/{stage}"
        start = time.time()
        if persist_checkpoint:
            df.write_parquet(uri)
        else:
            df.collect()
        end = time.time()
        logger.info(f"Finished {stage} stage in {end-start} sec")
        
        # Read Saved Checkpoint if needed
        if persist_checkpoint:
            df = daft.read_parquet(uri)
        
        return df

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
            .with_column(self.content_col, extract_blocks(col("content_raw")).list.join(" "))
            .with_column(self.index_col, monotonically_increasing_id())
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
        2: if l_v ≤ l_u then: # (if u < v)
        3:     Emit <u; v>
        4: else
        5:     Emit <v; u>

        7: Reduce hu;N ⊆ Γ(u)i:
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
            .select((col("u") > col("v")).if_else(ee(col("u"), col("v")), ee(col("v"), col("u"))).alias("e"))
            .select(col("e")["*"]) 
            
            # small_star_reduce
            .groupby("u").agg_list("v")
            .with_column("min_edge", col("v").list.min())
            .with_column("min_edge", (col("u") <= col("min_edge")).if_else(col("u"), col("min_edge")))
            .explode("v")
            .where(col("v") > col("u"))   
            .with_column("e", ee(col("v"), col("min_edge")))

            # Clean up
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
    

    def connected_components(self, df: DataFrame):
        # Initialize b
        b = (
            df.select(col("left_edge").alias("u"), col("right_edge").alias("v"))
            .where(~col("u").is_null())
            .where(~col("v").is_null())
            .collect() # Materialize
        )    

        #b = self._canonicalize_edges(b)
        check = self.check_igraph(b)
        print(check)

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

    def check_igraph(self, df: DataFrame):
        import igraph as ig
        df = df.select(col("u").cast(daft.DataType.int64()), col("v").cast(daft.DataType.int64())).to_pandas()
        g = ig.Graph.DataFrame(df, directed=False)
        return {frozenset(c) for c in g.connected_components(mode="strong") if len(c) > 1}

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
    
    def partitioned_save(self, df: DataFrame, chunk_size: int, max_partitions: int):
        start_time = time.time()
        df = df.collect()
        total_rows = df.count_rows()
        partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))

        (
            df #.repartition(partitions)
            #.with_column("__pid__", monotonically_increasing_id() / lit(2**36))
            .write_parquet(self.output_uri) #, partition_cols=["__pid__"], write_mode="overwrite", compression="snappy")
        )
        end_time = time.time()
        logger.info(f"Partitioned Saved {total_rows} rows in {end_time - start_time:.2f}s")
    
    def log_results(self, prepped: DataFrame, results: DataFrame, start_time: float, end_time: float):
        prepped_rows = prepped.count_rows()
        results_rows = results.count_rows()
        print("─" * 80)
        print(f"# of rows before:  {prepped_rows}")
        print(f"# of rows after:   {results_rows}")
        print(f"% of rows kept:    {results_rows / max(0, prepped_rows) * 100:.2f}%")
        print(f"Output Directory:  {self.output_uri}")
        print(f"Overall Time:      {end_time - start_time:.2f}s")
        print("─" * 80)

        



if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

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
    cc_search_pattern = "s3://commoncrawl/crawl-data/CC-MAIN-2024-42/segments/*/warc/*.warc.gz"
    base_uri = "/Users/everett-founder/git/ugh/daft-minhash-dedupe/data"
    row_limit = 100
    chunk_size = 200_000
    max_partitions = 2048
    persist_checkpoint = True
    remove_punct = False
    lowercase = False
    nfd_unicode = True
    white_space = True
    num_perm = 64
    ngram_size = 5
    seed = 42
    hash_function = 'xxhash'
    threshold = 0.717

    # Initialize Pipeline
    dedupe_pipeline = CommonCrawlHtmlMinHashDedupe(
        output_uri=f"{base_uri}/output",
        checkpoint_uri=f"{base_uri}/checkpoint",
    )

    # Run it
    results = dedupe_pipeline(
        cc_warc_uri=cc_search_pattern,
        row_limit=row_limit,
        chunk_size=chunk_size,
        max_partitions=max_partitions,
        persist_checkpoint=persist_checkpoint,
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