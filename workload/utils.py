import daft
import time
import math
from daft import lit, DataFrame
from daft.functions import monotonically_increasing_id
from scipy.integrate import quad as integrate
import ray

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


d

def log_results(output_uri: str, prepped: DataFrame, results: DataFrame, start_time: float, end_time: float):
    prepped_rows = prepped.count_rows()
    results_rows = results.count_rows()
    print("─" * 80)
    print(f"# of rows before:  {prepped_rows}")
    print(f"# of rows after:   {results_rows}")
    print(f"% of rows kept:    {results_rows / max(1, prepped_rows) * 100:.2f}%")
    print(f"Output Directory:  {output_uri}")
    print(f"Overall Time:      {end_time - start_time:.2f}s")
    print("─" * 80)

        
__all__ = ["optimal_param", "checkpoint", "partitioned_save", "log_results"]