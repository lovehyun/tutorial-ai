# sequence_length = 15일 때의 LSTM 이진 분류

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

data = np.load("lstm_data_seq15.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)

# 2. 정규화 (StandardScaler는 2D 입력 필요)
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)  # (samples * 5, 4)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

X = X_scaled.reshape(n_samples, seq_len, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save("model_lstm_seq15.h5")
joblib.dump(scaler, "scaler_seq15.pkl")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
