import pandas as pd

df = pd.read_parquet("large_test_file.parquet", engine="fastparquet")