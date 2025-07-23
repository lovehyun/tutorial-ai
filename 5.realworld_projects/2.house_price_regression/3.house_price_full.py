# california_housing_ml_workflow.py

# 1. 데이터 불러오기
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# 2. 데이터 전처리 - 정규화(StandardScaler)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. 데이터 분할 (학습용/테스트용)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 모델 학습
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# 5. 모델 저장
import joblib

joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(dt_model, "dt_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(gb_model, "gb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 6. 모델 앙상블
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error

# Voting Regressor (평균 기반)
voting_reg = VotingRegressor(estimators=[
    ('lr', lr_model),
    ('rf', rf_model),
    ('gb', gb_model)
])
voting_reg.fit(X_train, y_train)
voting_pred = voting_reg.predict(X_test)
print("Voting Regressor MSE:", round(mean_squared_error(y_test, voting_pred), 2))

# Stacking Regressor (메타 모델: LinearRegression)
stack_reg = StackingRegressor(
    estimators=[('dt', dt_model), ('rf', rf_model), ('gb', gb_model)],
    final_estimator=LinearRegression(),
    cv=5
)
stack_reg.fit(X_train, y_train)
stack_pred = stack_reg.predict(X_test)
print("Stacking Regressor MSE:", round(mean_squared_error(y_test, stack_pred), 2))
