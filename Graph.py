import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from faker import Faker

# Settings
fake = Faker()
np.random.seed(42)
n = 100_000

# Generate sample data
names = ['John' if i % 10 == 0 else fake.first_name() for i in range(n)]
ages = np.random.randint(18, 70, size=n)

df = pd.DataFrame({
    'name': names,
    'age': ages
})

# --- Performance Test Functions ---

def run_loop():
    result = []
    for i in range(len(df)):
        if df.iloc[i]['name'] == 'John' and df.iloc[i]['age'] > 25:
            result.append(df.iloc[i])
    return result

def run_vectorized():
    return df[(df['name'] == 'John') & (df['age'] > 25)]

# --- Timing ---
loop_times = []
vector_times = []

for _ in range(3):
    # Loop timing
    t0 = time.time()
    _ = run_loop()
    t1 = time.time()
    loop_times.append(t1 - t0)

    # Vector timing
    t0 = time.time()
    _ = run_vectorized()
    t1 = time.time()
    vector_times.append(t1 - t0)

# --- Plot Results ---
methods = ['Loop', 'Vectorized']
avg_times = [np.mean(loop_times), np.mean(vector_times)]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, avg_times, color=['red', 'green'])
plt.title("Performance Comparison: Loop vs Vectorized Search")
plt.ylabel("Average Time (seconds)")
plt.ylim(0, max(avg_times) * 1.2)

# Add labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.3, yval + 0.01, f"{yval:.3f}", fontsize=12)

plt.tight_layout()
plt.show()