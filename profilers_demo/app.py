import time
import math
import random

def cpu_heavy():
    print("Starting CPU-heavy task...")
    total = 0
    for i in range(1, 10**6):
        total += math.sqrt(i)
    print("CPU-heavy task complete.")
    return total

def io_heavy():
    print("Starting IO-heavy task...")
    for _ in range(5):
        time.sleep(0.5)
    print("IO-heavy task complete.")
    return "Done"

def mixed_work():
    print("Starting mixed task...")
    result = []
    for _ in range(10):
        result.append(cpu_heavy())
        time.sleep(0.1)
    print("Mixed task complete.")
    return result

def main():
    cpu_heavy()
    io_heavy()
    mixed_work()

if __name__ == "__main__":
    main()

