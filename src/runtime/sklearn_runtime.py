from __future__ import annotations

import os
import sys
import warnings


_KMEANS_MKL_WARNING_PATTERN = (
    r"KMeans is known to have a memory leak on Windows with MKL, "
    r"when there are less chunks than available threads\."
)


def configure_sklearn_runtime(*, omp_num_threads: int = 5) -> None:
    """Apply deterministic sklearn runtime defaults for Windows local runs."""
    if sys.platform.startswith("win"):
        os.environ.setdefault("OMP_NUM_THREADS", str(int(omp_num_threads)))
    warnings.filterwarnings(
        "ignore",
        message=_KMEANS_MKL_WARNING_PATTERN,
        category=UserWarning,
    )

