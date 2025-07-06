import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 샘플 데이터 생성 (비선형)
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 5 * np.sin(X) + np.random.randn(100, 1) * 0.5  # 비선형 데이터

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 결정 트리 모델 생성 및 학습
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE (평균 절대 오차): {mae:.2f}")
print(f"MSE (평균 제곱 오차): {mse:.2f}")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")
print(f"R² (설명력): {r2:.2f}")

# 6. 예측 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('결정 트리 회귀: 실제 값 vs 예측 값')
plt.legend()
plt.grid()
plt.show()

# 7. 전체 곡선 시각화
X_full = np.linspace(-3, 3, 100).reshape(-1, 1)
y_full_pred = model.predict(X_full)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Actual')
plt.plot(X_full, y_full_pred, color='red', label='Decision Tree Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('결정 트리 회귀: 전체 곡선')
plt.legend()
plt.grid()
plt.show()
