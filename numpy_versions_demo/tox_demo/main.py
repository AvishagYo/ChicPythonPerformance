import numpy as np
import time

print("NumPy version:", np.__version__)

N = 3000  # Large enough to show performance differences

# Generate large random matrices
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Time the matrix multiplication
start = time.time()
C = A @ B
end = time.time()

print(f"Matrix multiplication of {N}x{N} took: {end - start:.4f} seconds")
