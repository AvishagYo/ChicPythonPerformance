import csv
import random
import string

def generate_large_csv(file_path, target_size_mb=1000):
    row_size_bytes = 200  # Estimated row size
    target_size_bytes = target_size_mb * 1024 * 1024
    total_rows = target_size_bytes // row_size_bytes

    num_columns = 10
    column_names = [f"col_{i}" for i in range(num_columns)]

    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)

        for _ in range(int(total_rows)):
            row = [
                ''.join(random.choices(string.ascii_letters + string.digits, k=20))
                for _ in range(num_columns)
            ]
            writer.writerow(row)

    print(f"CSV file generated: {file_path}")

# Example usage
generate_large_csv("large_test_file.csv", target_size_mb=100)
