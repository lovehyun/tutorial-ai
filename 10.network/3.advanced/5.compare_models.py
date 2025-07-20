# sudo apt-get install fonts-nanum

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'

# 1. 데이터 로드 및 레이블 처리 (이진 분류: 0=정상, 1=비정상)
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)

# 2. 정규화: StandardScaler 적용
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, seq_len, n_features)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 실험 모델 구성 정의
experiments = {
    "LSTM": lambda: Sequential([
        LSTM(64, input_shape=(seq_len, n_features)),
        Dense(1, activation='sigmoid')
    ]),
    "LSTM + Dropout": lambda: Sequential([
        LSTM(64, input_shape=(seq_len, n_features)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ]),
    "GRU": lambda: Sequential([
        GRU(64, input_shape=(seq_len, n_features)),
        Dense(1, activation='sigmoid')
    ])
}

results = {}

# 5. 모델 훈련 및 평가
for name, build_model in experiments.items():
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

    y_prob = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    results[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score,
        "y_pred": (y_prob > 0.5).astype(int)
    }

# 6. ROC Curve 시각화 (한글 폰트 설정 포함)
# plt.rcParams['font.family'] = 'NanumGothic'  # 한글폰트 설치 필요
plt.figure(figsize=(8, 6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['auc']:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("LSTM 기반 모델 ROC 곡선 비교")
plt.xlabel("거짓 양성 비율 (False Positive Rate)")
plt.ylabel("진짜 양성 비율 (True Positive Rate)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. 정량 평가 출력
for name, res in results.items():
    print(f"\n{name} 분류 리포트:")
    print(classification_report(y_test, res["y_pred"]))
