# Autoencoder 기본 구조
#  입력         잠재공간         출력
#   X   ─▶ [인코더] ─▶ Z ─▶ [디코더] ─▶ X'
# X: 원본 입력 데이터
# Z: 인코더에 의해 축소된 특징 공간 (latent space)
# X': 디코더가 복원한 출력
# 손실: loss = ||X - X'||² (복원 오류)

# Autoencoder는 주로 정상 데이터만 학습시킵니다.
# 이후 테스트할 때:
#  - 정상 입력은 잘 복원됨 → 손실 작음
#  - 이상 입력은 잘 복원 못함 → 손실 큼
# 그래서 복원 오차가 큰 데이터를 **이상(anomaly)**으로 간주합니다.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import matplotlib.pyplot as plt
import matplotlib
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')  # Windows
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')    # macOS
else:
    matplotlib.rc('font', family='NanumGothic')    # Linux (Nanum 설치 필요)

# 마이너스 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False


# 1. 정상 데이터 1000개, 이상 데이터 100개 생성 (예시)
np.random.seed(42)
normal = np.column_stack([
    np.random.normal(20, 4, 1000),     # duration
    np.random.normal(200, 20, 1000),   # packet_size
    np.random.normal(500, 50, 1000),   # src_bytes
    np.random.normal(600, 50, 1000),   # dst_bytes
])
anomaly = np.column_stack([
    np.random.normal(60, 10, 100),     # 이상하게 긴 duration
    np.random.normal(600, 50, 100),    # 비정상적으로 큰 packet
    np.random.normal(1000, 100, 100),
    np.random.normal(1200, 100, 100),
])

X_all = np.vstack([normal, anomaly])
y_all = np.array([0]*1000 + [1]*100)  # 0=정상, 1=비정상

# 2. 정규화
# feature의 값 범위가 다르므로 정규화를 통해 모델 학습을 쉽게 만듦
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# 3. train/test 분리 (train에는 정상만)
# 학습에는 정상 데이터만 사용 (이게 포인트!)
# 테스트에는 정상 + 비정상 모두 포함해서 얼마나 잘 구분하는지 확인
X_train = X_scaled[y_all == 0]  # 정상만 학습
X_test = X_scaled               # 전체 테스트
y_test = y_all

# 4. Autoencoder 모델
# 입력 → 압축(인코더) → 복원(디코더)
# 복원한 값이 얼마나 원래와 다른지(MSE)로 학습
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)
autoencoder = Model(input_layer, decoded) # 정상 데이터로 Autoencoder 학습
autoencoder.compile(optimizer="adam", loss="mse")

# 5. 학습
autoencoder.fit(X_train, X_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

# 6. 복원값 계산
X_pred = autoencoder.predict(X_test)
recon_error = np.mean((X_test - X_pred)**2, axis=1)

# 7. 이상 탐지 (임계값 기준)
threshold = np.percentile(recon_error, 95)
y_pred = (recon_error > threshold).astype(int)

# 8. 평가
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. 시각화 (선택)
plt.hist(recon_error[y_test == 0], bins=50, alpha=0.7, label="정상")
plt.hist(recon_error[y_test == 1], bins=50, alpha=0.7, label="비정상")
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.3f}")
plt.legend()
plt.title("복원 오류 분포")
plt.xlabel("재구성 오차")
plt.ylabel("빈도")
plt.show()
