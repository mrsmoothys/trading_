import pandas as pd
feature_importance = pd.read_csv('/Users/mrsmoothy/Desktop/rsidtrade/trading_/results/feature_importance_20250317_224706.csv')
print(feature_importance.sort_values('importance', ascending=False))