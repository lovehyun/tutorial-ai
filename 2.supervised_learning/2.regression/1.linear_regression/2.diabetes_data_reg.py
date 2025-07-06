import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 데이터 로드 (당뇨병 데이터셋 사용)
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. 데이터 분할 (학습/테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 데이터 표준화 (선형 회귀에서 필수는 아니지만, 데이터 안정성을 위해 사용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 선형 회귀 모델 생성 및 학습
lr = LinearRegression()
lr.fit(X_train, y_train)

# 5. 예측
y_pred = lr.predict(X_test)

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
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('선형 회귀: 실제 값 vs 예측 값')
plt.grid()
plt.show()
