from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 데이터 로드
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 지금 이 시점에서 저장 가능
import joblib
joblib.dump(model, "rf_model.pkl")

# 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# 스케일링 필요성:
# - 랜덤 포레스트는 결정 트리 기반 모델이기 때문에...
# - 각 특성(feature)의 값의 상대 크기보다,
#   특정 값 기준으로 분할(split) 하는 방식에 더 관심이 있습니다.
# 즉, 10, 1000, 0.01 같이 크기가 매우 달라도
# - 분할 조건에 따라 나누기 때문에
# - 스케일(정규화, 표준화)은 크게 영향을 주지 않습니다.

# | 모델 종류              | 스케일링 필요성             |
# | --------------------- | -------------------------- |
# | SVM (`SVC`)           | 🔴 매우 필요               |
# | KNN                   | 🔴 매우 필요               |
# | 로지스틱 회귀          | 🔴 필요                    |
# | 선형 회귀              | 🔴 필요                    |
# | Naive Bayes           | 🔴 보통 필요                |
# | 결정 트리 / 랜덤 포레스트    | 🟢 불필요              |
# | XGBoost / LightGBM    | 🟡 대부분 불필요 (하지만 해도 무방) |


# Q. Linear Regression으로 Iris 문제를 풀 수 있을까? 아니오. 적합하지 않습니다.
# Iris 문제는 꽃 품종 분류 문제입니다.
# 즉, 정답이 숫자가 아니라 범주형(0, 1, 2 → setosa, versicolor, virginica) 입니다.
# 따라서 회귀가 아니라 분류(classification) 문제입니다.
#
# 적합한 모델:
# - LogisticRegression (이름에 Regression 들어있지만 분류 모델임)
# - RandomForestClassifier
# - SVC 등
