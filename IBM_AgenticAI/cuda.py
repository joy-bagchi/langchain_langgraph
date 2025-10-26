import cupy as cp
import numpy as np
import numba as nb
from numba import cuda
from numba.cuda import cuda_error, is_available

cp.random.seed(42)
print(cp.is_available())
# Run an expensive computation on the GPU
result = cp.random.normal(size=(100000, 10000))
print(result)

