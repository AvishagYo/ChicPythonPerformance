from benchmark import perf
from random import random
import numpy as np

print("calling regular")
DATA = [random() for _ in range(30_000_000)]

with perf():
    mean = sum(DATA) / len(DATA)
    result = [DATA[i] - mean for i in range(len(DATA))]


print("calling numpy")
NUMPY_DATA = np.random.rand(30_000_000)

with perf():
    mean = NUMPY_DATA.sum() / len(NUMPY_DATA)
    result = NUMPY_DATA - mean