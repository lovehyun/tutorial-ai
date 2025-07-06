# 로지스틱 회귀는 '이름'은 회귀지만, '목적'은 분류입니다.
# 왜 이름은 Regression인가?
# - 로지스틱 회귀도 **"수학적으로는 회귀식"**을 사용합니다.
# - 회귀처럼 y = b0 + b1 * x 형태로 선형 조합을 만들어요.
# - 그런데 결과를 그대로 쓰지 않고 시그모이드 함수를 적용합니다.

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 이진 분류용 더미 데이터 생성
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 정확도 출력
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")
