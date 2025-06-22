from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 로드
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", round(mse, 2))
