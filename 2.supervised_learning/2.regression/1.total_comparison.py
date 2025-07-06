# 비교 파일	    데이터셋	    비교 모델	          평가 지표
# 회귀	        Diabetes	LR, DT, RF, GB, SVR	    MAE, MSE, R²

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 로드
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 데이터 스케일링 (필요한 모델: SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 목록
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR()
}

# 5. 모델 평가
results = []

for name, model in models.items():
    # SVR은 스케일링 필수
    if name == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((name, mae, mse, r2))

# 6. 결과 출력
results.sort(key=lambda x: x[2])  # MSE 기준 오름차순 정렬

print("\n회귀 모델 성능 비교 (MSE 기준):")
for name, mae, mse, r2 in results:
    print(f"{name}: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.4f}")
