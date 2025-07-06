# 개념
# - 서포트 벡터 머신(SVM)의 회귀 버전
# - 두 경계선 사이의 마진 안에 최대한 많은 데이터를 포함시키는 방식
# - 복잡한 데이터에 강하지만 계산이 오래 걸릴 수 있음
# - 스케일링 필수
#
# 핵심 포인트
# --------------------
# 항목	        설명
# 데이터	    비선형 (y = 5sin(x) + noise)
# 스케일링	    필수 (입력/출력 모두)
# 주요 파라미터	    kernel='rbf', C=10, epsilon=0.1
# 특징	        계산 복잡, 고성능, 마진 기반 예측

# 추가 실습
# - kernel='linear', kernel='poly'로 변경하고 성능 비교
# - C 값 (규제 강도)와 epsilon (허용 오차) 조정 실습

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 샘플 데이터 생성 (비선형)
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 5 * np.sin(X) + np.random.randn(100, 1) * 0.5  # 비선형 데이터

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 데이터 스케일링 (SVR은 스케일링 필수)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 4. SVR 모델 생성 및 학습
model = SVR(kernel='rbf', C=10, epsilon=0.1)
model.fit(X_train_scaled, y_train_scaled.ravel())

# 5. 예측
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# 6. 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE (평균 절대 오차): {mae:.2f}")
print(f"MSE (평균 제곱 오차): {mse:.2f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")
print(f"R² (설명력): {r2:.2f}")

# 7. 예측 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('SVR: 실제 값 vs 예측 값')
plt.legend()
plt.grid()
plt.show()

# 8. 전체 곡선 시각화
X_full = np.linspace(-3, 3, 100).reshape(-1, 1)
X_full_scaled = scaler_X.transform(X_full)
y_full_pred_scaled = model.predict(X_full_scaled)
y_full_pred = scaler_y.inverse_transform(y_full_pred_scaled.reshape(-1, 1))

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Actual')
plt.plot(X_full, y_full_pred, color='red', label='SVR Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('SVR: 전체 곡선')
plt.legend()
plt.grid()
plt.show()
