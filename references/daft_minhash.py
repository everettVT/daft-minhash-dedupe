#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "daft",
#     "scipy",
# ]
# ///

from __future__ import annotations

import argparse
import logging
import math
import time
import warnings

import daft
from daft import DataFrame, Expression, col, lit, struct
from daft.functions import monotonically_increasing_id

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from scipy.integrate import quad as integrate

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SEED = 42
MAX_WRITE_CHUNK_SIZE: int = 200_000
MAX_WRITE_PARTITIONS: int = 2048


# region: Connected Components
def ee(u: Expression, v: Expression):
    return struct(u.alias("u"), v.alias("v"))

def components(
    df: DataFrame,
    left_id_col: str = "u",
    right_id_col: str = "v",
    output_index_col: str = "u",
    output_component_col: str = "component"
) -> DataFrame:
    b = (
        df.select(col(left_id_col).alias("u"), col(right_id_col).alias("v"))
        .where(~col("u").is_null())
        .where(~col("v").is_null())
        .collect()
    )    
    while True:
        a = (b
             # large_star_map
             .select("u", "v")
             .union_all(b.select(col("v").alias("u"), col("u").alias("v")))
             .groupby("u").agg_list("v")

             # large_star_reduce
             .with_column("min_edge", col("v").list.min()) # Get minimum of v neighbors
             .with_column("min_edge", (col("u") <= col("min_edge")).if_else(col("u"), col("min_edge"))) # Get minimum of u and min_edge
             .select(
                col("u").list.map(
                    ee(daft.element(), col("min_edge")).alias("e")
                ),
                col("u")
            )

             .explode("e")
             .where(col("e")["v"] > col("u")).select("e")
             .where(~col("e").is_null())
             .distinct()
             .select(col("e")["*"])
             .where(col("u") != col("v"))
             .collect()
        )
        b = (a
             # small_star_map
             .select((col("u") > col("v")).if_else(ee(col("u"), col("v")), ee(col("v"), col("u"))).alias("e"))
             .select(col("e")["*"])

             .groupby("u").agg_list("v")
             # small_star_reduce
             .with_column("min_edge", col("v").list.min())
             .with_column("min_edge", (col("u") <= col("min_edge")).if_else(col("u"), col("min_edge")))
             .select(col("u").list.map(ee(daft.element(), col("min_edge"))).alias("e"), col("u"), col("min_edge"))
             # TODO: list_append

             .explode("e")
             .where(~col("e").is_null())
             .distinct()
             .select(col("e")["*"])
             .collect()
        )
        # check convergence
        a_hash = a.select(col("u").hash().alias("hash")).sum("hash").to_pydict()["hash"][0]
        b_hash = b.select(col("u").hash().alias("hash")).sum("hash").to_pydict()["hash"][0]
        if a_hash == b_hash:
            return (
                b
                .select(col("u").alias(output_index_col), col("v").alias(output_component_col))
                .collect()
            )
# endregion

# region: MinHashLSH
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

def partitioned_save(df: DataFrame, chunk_size: int, max_partitions: int, output: str):
    df = df.collect()
    total_rows = df.count_rows()
    partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))

    (
        df.repartition(partitions)
        .with_column("__pid__", monotonically_increasing_id() / lit(2**36))
        .write_parquet(output, partition_cols=["__pid__"], write_mode="overwrite", compression="snappy")
    )


if __name__ == "__main__":
    daft.context.set_runner_ray()
    DEFAULT_INDEX = "__id__"

    # region: Argument Parsing
    parser = argparse.ArgumentParser(description="Intra-dataset near-deduplicating with daft")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory of Parquet files")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory of Parquet files")

    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument("--bands", "-b", type=int, default=None, help="Number of bands")
    parser.add_argument("--rows", "-r", type=int, default=None, help="Number of rows per band")
    parser.add_argument("--num_perm", type=int, default=250, help="Number of permutations")
    parser.add_argument("--column", "-c", type=str, default="content", help="Column to deduplicate on")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX, help="Column to index on")
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum token length of document to be considered. Short ones will be removed",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode by saving cluster results",
    )

    args = parser.parse_args()
    input_dir: str = args.input
    output_dir: str = args.output
    checkpoint_dir: str = args.checkpoint_dir
    threshold: float = args.threshold
    ngram_size: int = args.ngram_size
    bands: int | None = args.bands
    rows: int | None = args.rows
    num_perm: int = args.num_perm
    content_col: str = args.column
    index_col: str = args.index
    min_length: int = args.min_length
    debug: bool = args.debug
    # endregion

    B, R = args.bands, args.rows
    if B is None or R is None:
        B, R = optimal_param(args.threshold, args.num_perm)

    # region: Print Inputs & Arguments
    print("─" * 120)
    print(f"Using {num_perm=} hashes: {B} bands, {R} hashes per band")
    print(f"{input_dir=}")
    print(f"{output_dir=}")
    print(f"{threshold=}")
    print(f"{ngram_size=}")
    print(f"{min_length=}")
    print(f"{content_col=}")
    print(f"{index_col=}")
    print("─" * 120)

    # region: Load & Preprocess Data
    start_time = time.time()
    df = (
        daft.read_parquet(input_dir)
        .with_column(content_col, col(content_col).cast(str))
        .filter((col(content_col).str.split(r"\W", regex=True).list.count()) >= min_length)
    )
    if index_col == DEFAULT_INDEX:
        df = df.with_column(DEFAULT_INDEX, monotonically_increasing_id())
    else:
        df = df.with_column(index_col, col(index_col).cast(int))
    df.write_parquet(f"{checkpoint_dir}/data")
    print(f"Pre-Process Time: {time.time() - start_time:.2f}s")
    df = daft.read_parquet(f"{checkpoint_dir}/data")
 
    DATA_SIZE = df.count_rows()
    if DATA_SIZE == 0:
        print("No data found.")
        exit(0)

    print(f"# of Documents: {DATA_SIZE}")
    print(df.schema())
    print("─" * 120)
    # endregion

    # region: MinHash
    edges_start_time = time.time()
    edges = (
        df
        .select(index_col, content_col)

        # MinHash Generation
        .with_column(content_col, col(content_col).str.normalize(white_space=True))
        .with_column("min_hashes", col(content_col).minhash(
            num_hashes=num_perm,
            ngram_size=ngram_size,
            seed=SEED,
            hash_function="xxhash",
        ))
        # Band Generation
        .with_column("bands", col("min_hashes").list.chunk(R))
        .with_column("band_idxs", lit(list(range(B))))
        .explode("bands", "band_idxs")
        .select(index_col, band_idx=col("band_idxs"), band=col("bands"))
        # Grouping Bands
        .groupby(col("band_idx"), col("band"))
        .agg(col(index_col).agg_list().alias("nodes"))
        # Generate Graph Edges
        .with_column("left_edge", col("nodes").list.min())
        .explode("nodes")
        .select("left_edge", right_edge=col("nodes"))
        .filter(col("left_edge") != col("right_edge"))
        .distinct()
        # Checkpoint
        .write_parquet(f"{checkpoint_dir}/edges")
    )
    edges = daft.read_parquet(f"{checkpoint_dir}/edges")
    print(f"Generate Edges in {time.time() - edges_start_time:.2f}s")  # and {edges.count_rows()} rows"
    # endregion

    # region: Connected Components
    if edges.count_rows() == 0:
        partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, output_dir)
        
        print("─" * 120)
        print("No duplicates found.")
        print(f"Data Output:    {output_dir}")
        print(f"Time:           {time.time() - start_time:.2f}s")
        print("─" * 120)

    # Actual connected components
    assignment_start_time = time.time()
    assignment = components(
        edges, left_id_col="left_edge", right_id_col="right_edge", output_index_col=index_col, output_component_col="__component__"
    )
    assignment.write_parquet(f"{checkpoint_dir}/assignment")
    assignment = daft.read_parquet(f"{checkpoint_dir}/assignment")
    print(f"Generate Assignments in {time.time() - assignment_start_time:.2f}s") # and {assignment.count_rows()} rows")
    # endregion

    if debug:
        # save assignment for debugging purposes
        assignment.write_parquet(f"{output_dir}-assignment.parquet", write_mode="overwrite")

    # region: Merge Results
    # justification: this is needed for final output
    merge_start_time = time.time()
    df = df.into_batches(100_000)
    df = df.join(
        assignment.select(col(index_col), col("__component__")).repartition(),
        on=index_col,
        how="left",
    )
    df = (
        df
        .filter(col("__component__").is_null() | (col("__component__") == col(index_col)))
        .exclude("__component__")
    )
    df.explain(show_all=True)
    FINAL_SIZE = df.count_rows()
    
    df.write_parquet(output_dir, write_mode="overwrite", compression="snappy")
    print(f"Merging Time: {time.time() - merge_start_time:.2f}s")
    # endregion

    # region: Print Results
    print("─" * 120)
    print(f"# of rows before:  {DATA_SIZE}")
    print(f"# of rows after:   {FINAL_SIZE}")
    print(f"% of rows kept:    {FINAL_SIZE / max(0, DATA_SIZE) * 100:.2f}%")
    print(f"Output Directory:  {output_dir}")
    print(f"Overall Time:      {time.time() - start_time:.2f}s")
    print("─" * 120)
    # endregion