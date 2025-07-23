from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 데이터 로드
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 3. 모델 저장
import joblib
joblib.dump(model, "lr_model.pkl")

# 4. 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", round(mse, 2))


# LinearRegression은 **계수(가중치)**를 학습하는 모델입니다.
# 이때, 각 **피처의 스케일(값의 크기)**이 다르면,
# → 특정 피처가 불균형하게 큰 영향을 미칠 수 있습니다.
#
# 예: AveRooms는 평균 방 개수로 값이 작고,
#     Population은 인구 수로 수천 단위일 수 있음.

# Q. Linear Regression으로 California Housing 문제를 풀 수 있을까?
# A. 네, 매우 적합합니다.
#   정답(target) 이 실수값(중간 주택 가격) 이므로 → 회귀(regression) 문제입니다.
#   LinearRegression, Ridge, Lasso, GradientBoostingRegressor, RandomForestRegressor 등 모두 사용 가능합니다.
