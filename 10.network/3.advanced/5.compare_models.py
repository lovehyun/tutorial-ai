import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split

# 데이터 로드 (sequence_length = 10 기준)
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)  # 이진 분류

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

results = {}

# 실험 조합 정의
experiments = {
    "LSTM": lambda: Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1, activation='sigmoid')
    ]),
    "LSTM + Dropout": lambda: Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ]),
    "GRU": lambda: Sequential([
        GRU(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1, activation='sigmoid')
    ])
}

# 모델 훈련 및 평가 루프
for name, build_model in experiments.items():
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_prob = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    results[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score,
        "y_pred": (y_prob > 0.5).astype(int)
    }

# ROC Curve 시각화
plt.figure(figsize=(8, 6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['auc']:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve Comparison (LSTM Models)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# 정량 평가 출력
for name, res in results.items():
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, res["y_pred"]))
