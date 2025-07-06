import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 샘플 데이터 생성 (곡선 형태)
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 5 * (X ** 2) + np.random.randn(100, 1) * 5 + 10  # 비선형 관계

# 2. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 다항 특성 생성 (2차 다항식)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 4. 데이터 표준화
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

# 5. 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. 예측
y_pred = model.predict(X_test_poly)

# 7. 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE (평균 절대 오차): {mae:.2f}")
print(f"MSE (평균 제곱 오차): {mse:.2f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")
print(f"R² (설명력): {r2:.2f}")

# 8. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('다항 회귀: 실제 값 vs 예측 값')
plt.legend()
plt.grid()
plt.show()

# 9. 전체 곡선 시각화
X_full = np.linspace(-3, 3, 100).reshape(-1, 1)
X_full_poly = scaler.transform(poly.transform(X_full))
y_full_pred = model.predict(X_full_poly)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Actual')
plt.plot(X_full, y_full_pred, color='red', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('다항 회귀: 전체 곡선')
plt.legend()
plt.grid()
plt.show()
