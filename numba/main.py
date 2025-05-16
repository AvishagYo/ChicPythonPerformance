from time import time
import numpy as np
from numba import njit
import random

# Pure Python version:
def mean_distance_from_zero(arr):
    total = 0
    for i in range(len(arr)):
        total += abs(arr[i])
    return total / len(arr)

# A fast, JITed version:
mean_distance_from_zero_numba = njit(
    mean_distance_from_zero
)

arr = np.array(random.sample(range(1, 10_000_000), 10_000), dtype=np.float64)

start = time()
mean_distance_from_zero(arr)
print("Elapsed CPython: ", time() - start)

for i in range(10):
    start = time()
    mean_distance_from_zero_numba(arr)
    print("Elapsed Numba:   ", time() - start)