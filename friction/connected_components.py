from daft import col, struct, DataFrame


def _ee(u, v):
    return struct(u.alias("u"), v.alias("v"))


class ConnectedComponents:
    """Star-contraction connected components with robust final labeling.

    API
    - run(df_bands): accepts an LSH banding result with column "nodes" (list of node ids).
      Returns a DataFrame with columns ["u", "rep"] mapping each node to its component representative.
    - compute_from_edges(edges): accepts an edge list DataFrame with columns ["u", "v"].
    """

    # Edge utilities -----------------------------------------------------------
    def _build_edges(self, df: DataFrame) -> DataFrame:
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
        return (
            df
            .select((col("u") <= col("v")).if_else(_ee(col("u"), col("v")), _ee(col("v"), col("u"))).alias("e"))
            .select(col("e")["*"])
            .where(col("u") != col("v"))
            .distinct()
            .collect()
        )

    def _symmetrize(self, df: DataFrame) -> DataFrame:
        return (
            df
            .select("u", "v")
            .union_all(df.select(col("v").alias("u"), col("u").alias("v")))
            .collect()
        )

    # Star steps ---------------------------------------------------------------
    def _large_star(self, edges: DataFrame) -> DataFrame:
        E = self._symmetrize(edges)
        neigh = (
            E
            .groupby("u").agg_list("v")
            .with_column("nbrs", col("v"))
            .with_column("m", col("nbrs").list.min())
            .with_column("m", (col("u") < col("m")).if_else(col("u"), col("m")))
        )
        out = (
            neigh.explode("nbrs")
                .where(col("nbrs") > col("u"))
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .where(col("u") != col("v"))
                .distinct()
                .collect()
        )
        return out

    def _small_star(self, edges: DataFrame) -> DataFrame:
        directed = self._canonicalize_edges(edges)
        neigh = (
            directed
            .groupby("u").agg_list("v")
            .with_column("nbrs", col("v"))
            .with_column("m", col("nbrs").list.min())
            .with_column("m", (col("u") < col("m")).if_else(col("u"), col("m")))
        )
        out = (
            neigh.explode("nbrs")
                .select(col("nbrs").alias("u"), col("m").alias("v"))
                .where(col("u") != col("v"))
                .distinct()
                .collect()
        )
        return out

    # Equality helpers ---------------------------------------------------------
    def _check_canonical_set_equality(self, a: DataFrame, b: DataFrame) -> bool:
        ca = self._canonicalize_edges(a)
        cb = self._canonicalize_edges(b)
        left_minus  = ca.join(cb, on=["u","v"], how="anti").count_rows()
        right_minus = cb.join(ca, on=["u","v"], how="anti").count_rows()
        return (left_minus == 0) and (right_minus == 0)

    def _pairs_equal(self, a: DataFrame, b: DataFrame) -> bool:
        left_minus  = a.join(b, on=["u","rep"], how="anti").count_rows()
        right_minus = b.join(a, on=["u","rep"], how="anti").count_rows()
        return (left_minus == 0) and (right_minus == 0)

    # Core algorithm -----------------------------------------------------------
    def compute_from_edges(self, edges: DataFrame) -> DataFrame:
        b = self._canonicalize_edges(edges)

        iter_num = 0
        max_iters = 100
        b_prev = None
        while True:
            iter_num += 1
            a = self._large_star(b)
            b_next = self._small_star(a)

            if (b_prev is not None) and self._check_canonical_set_equality(b_prev, b_next):
                b = b_next
                break
            if iter_num >= max_iters:
                b = b_next
                break

            b_prev = b
            b = b_next

        nodes = (
            b.select(col("u").alias("u"))
             .union_all(b.select(col("v").alias("u")))
             .distinct()
             .collect()
        )
        rep_map = (
            b
            .groupby("u").agg(col("v").min().alias("rep"))
            .collect()
        )
        assignments = (
            nodes
            .join(rep_map, on="u", how="left")
            .with_column("rep", col("rep").is_null().if_else(col("u"), col("rep")))
            .select("u", "rep")
            .distinct()
            .collect()
        )

        # Path compression
        pc_iters = 0
        pc_max_iters = 100
        while pc_iters < pc_max_iters:
            pc_iters += 1
            next_assignments = (
                assignments
                .join(
                    assignments.select(col("u").alias("u2"), col("rep").alias("rep_of_rep")),
                    left_on="rep",
                    right_on="u2",
                    how="left",
                )
                .with_column("rep", col("rep_of_rep").is_null().if_else(col("rep"), col("rep_of_rep")))
                .select("u", "rep")
                .distinct()
                .collect()
            )
            if self._pairs_equal(assignments, next_assignments):
                break
            assignments = next_assignments

        # Global-min label propagation
        E = self._symmetrize(b)
        labels = assignments.select(col("u"), col("rep").alias("label")).collect()
        lp_iters = 0
        lp_max_iters = 100
        while lp_iters < lp_max_iters:
            lp_iters += 1
            nbr_min = (
                E
                .join(labels, left_on="v", right_on="u", how="left")
                .select(col("u").alias("node"), col("label"))
                .groupby("node")
                .agg(col("label").min().alias("nbr_min"))
                .collect()
            )
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
                .collect()
            )
            if self._pairs_equal(
                assignments.select(col("u"), col("rep").alias("label")).select(col("u"), col("label").alias("rep")),
                labels_next.select(col("u"), col("label").alias("rep")),
            ):
                break
            assignments = labels_next.select(col("u"), col("label").alias("rep")).collect()
            labels = labels_next

        return assignments

    def run(self, df_bands: DataFrame) -> DataFrame:
        edges = self._build_edges(df_bands)
        return self.compute_from_edges(edges)


__all__ = ["ConnectedComponents"]


