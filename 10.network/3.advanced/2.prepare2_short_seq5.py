import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sequence_length = 5
features = ["duration", "packet_size", "src_bytes", "dst_bytes"]

df = pd.read_csv("network_sequence.csv")
df = df.sort_values("timestamp")

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X, y = [], []

for i in range(len(df) - sequence_length):
    seq_x = df[features].iloc[i:i+sequence_length].values
    seq_y = df["label"].iloc[i+sequence_length]
    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

np.savez("lstm_data_seq5.npz", X=X, y=y)
print("lstm_data_seq5.npz 저장 완료")
