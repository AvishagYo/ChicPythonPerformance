import pandas as pd

df = pd.read_csv("large_test_file.csv")
df.to_parquet("large_test_file.parquet", compression=None)