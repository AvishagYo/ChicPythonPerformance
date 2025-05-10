# script.py
import sys
import time

sys.setrecursionlimit(10**6)

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

N = 35  # Adjust for ~1-10s runtime

start = time.time()
result = fib(N)
duration = time.time() - start

print(f"fib({N}) = {result}")
print(f"Execution time: {duration:.4f} seconds")

