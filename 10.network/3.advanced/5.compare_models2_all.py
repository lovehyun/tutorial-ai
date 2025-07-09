import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

def load_data(npz_path, binary=True):
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    if binary:
        y = (y != 0).astype(int)
    return X, y

def train_model(X, y, model_fn, label, epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = model_fn(X.shape[1], X.shape[2])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "label": label,
        "model": model,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "y_test": y_test,
        "y_pred": y_pred
    }

# 모델 정의 함수들
def make_lstm(seq_len, feat_dim):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, feat_dim)))
    model.add(Dense(1, activation='sigmoid'))
    return model

def make_gru(seq_len, feat_dim):
    model = Sequential()
    model.add(GRU(64, input_shape=(seq_len, feat_dim)))
    model.add(Dense(1, activation='sigmoid'))
    return model

def make_lstm_dropout(seq_len, feat_dim):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, feat_dim)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 실험 조합 정의
experiments = [
    ("lstm_data_seq5.npz", make_lstm, "LSTM (seq=5)"),
    ("lstm_data.npz", make_lstm, "LSTM (seq=10)"),
    ("lstm_data_seq15.npz", make_lstm, "LSTM (seq=15)"),
    ("lstm_data.npz", make_gru, "GRU (seq=10)"),
    ("lstm_data.npz", make_lstm_dropout, "LSTM+Dropout (seq=10)")
]

# 실험 실행
results = []
for npz_path, model_fn, label in experiments:
    print(f"Training: {label}")
    X, y = load_data(npz_path)
    result = train_model(X, y, model_fn, label)
    results.append(result)

# ROC Curve 시각화
plt.figure(figsize=(8, 6))
for res in results:
    plt.plot(res["fpr"], res["tpr"], label=f"{res['label']} (AUC={res['auc']:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve Comparison (All LSTM Variants)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Classification Report 출력
for res in results:
    print(f"\n=== {res['label']} ===")
    print(classification_report(res["y_test"], res["y_pred"]))
