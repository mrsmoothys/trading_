import pandas as pd
data_path = "/Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv"
df = pd.read_csv(data_path)
print(df.head())
print(f"Loaded {len(df)} rows of data")