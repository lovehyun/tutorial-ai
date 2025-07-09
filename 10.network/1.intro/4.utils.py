# 4_utils.py
import pandas as pd

def load_data(file_path="network_data.csv"):
    return pd.read_csv(file_path)
