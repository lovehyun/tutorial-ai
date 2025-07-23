from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# 1. 데이터 로드
data = fetch_california_housing()
X, y = data.data, data.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 스케일링 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5. 모델 및 스케일러 저장
joblib.dump(model, "lr_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 6. 평가
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("MSE (스케일링 적용):", round(mse, 2))
