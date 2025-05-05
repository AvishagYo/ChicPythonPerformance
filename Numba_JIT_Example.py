import numpy as np
from numba import jit
import time

# non optimized function
def slow_function(x):
    total = 0
    for i in range(1000000):
        total += i * x
    return total

# optimized function - using @jit
@jit(nopython=True)
def fast_function(x):
    total = 0
    for i in range(1000000):
        total += i * x
    return total

# performance check
x = 2

# non optimized function runtime
start_time = time.time()
slow_function(x)
end_time = time.time()
print(f"Time for slow function: {end_time - start_time:.5f} seconds")

# using @jit function runtime
start_time = time.time()
fast_function(x)
end_time = time.time()
print(f"Time for JIT function With completion: {end_time - start_time:.5f} seconds")

# using @jit function runtime
start_time = time.time()
fast_function(x)
end_time = time.time()
print(f"Time for JIT function After completion: {end_time - start_time:.5f} seconds")