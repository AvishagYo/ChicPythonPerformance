from benchmark import perf
from random import random

print("calling pypy")
DATA = [random() for _ in range(30_000_000)]

with perf():
    mean = sum(DATA) / len(DATA)
    result = [DATA[i] - mean for i in range(len(DATA))]
