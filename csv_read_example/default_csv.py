#https://pythonspeed.com/articles/pandas-read-csv-fast/
import pandas as pd
import time
start_time = time.time()
df = pd.read_csv("large_test_file.csv")
end_time = time.time()

execution_time = end_time - start_time
print(f"Script execution time: {execution_time:.4f} seconds")