# Vectorization performance sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import time
from faker import Faker

from timeitt import timeitt

# Generate a DataFrame with 1 million random integers
n = 1_000_000
df1 = pd.DataFrame({
    'a': np.random.randint(1, 100, size=n),
    'b': np.random.randint(1, 100, size=n)
})

# Create fake people data
fake = Faker()
np.random.seed(42)

# Generate 100,000 records
m = 100_000
names = ['John' if i % 10 == 0 else fake.first_name() for i in range(m)]
ages = np.random.randint(18, 70, size=m)

df2 = pd.DataFrame({
    'name': names,
    'age': ages
})

@timeitt
def using_loop_calc():
    # Loop-based calculation: a^2 + b^2
    start_time = time.time()
    result_loop = []
    for i in range(len(df1)):
        result_loop.append(df1.iloc[i]['a'] ** 2 + df1.iloc[i]['b'] ** 2)
    end_time = time.time()
    print(f"Looping took: {end_time - start_time:.2f} seconds")

def using_vectorization_calc():
    # Vectorized calculation
    start_time = time.time()
    result_vectorized = df1['a'] ** 2 + df1['b'] ** 2
    end_time = time.time()
    print(f"Vectorized operation took: {end_time - start_time:.2f} seconds")

def arithmetic_comparison():
    using_loop_calc()
    using_vectorization_calc()

def using_loop_search():
    start_time = time.perf_counter()
    matches_loop = []
    for i in range(len(df2)):
        if df2.iloc[i]['name'] == 'John' and df2.iloc[i]['age'] > 25:
            matches_loop.append(df2.iloc[i])
    end_time = time.perf_counter()
    print(f"Loop took: {end_time - start_time:.2f} seconds")
    print(f"Found {len(matches_loop)} Johns over 25")

def using_vectorization_search():
    start_time = time.perf_counter()
    matches_vectorized = df2[(df2['name'] == 'John') & (df2['age'] > 25)]
    end_time = time.perf_counter()
    print(f"Vectorized took: {end_time - start_time:.2f} seconds")
    print(f"Found {len(matches_vectorized)} Johns over 25")

def search_comparison():
    using_loop_search()
    using_vectorization_search()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('Pycharm')
    print('----- Looping vs Vectorization -----')
    print('Arithmetic comparison:')
    arithmetic_comparison()
    print()
    print('Search comparison:')
    search_comparison()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
