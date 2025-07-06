# 파이프라인 (Pipeline) 은 머신러닝에서
# 👉 데이터 전처리부터 모델 훈련, 튜닝, 예측까지 전 과정을 하나로 묶어주는 자동화 도구입니다.

# ✅ 파이프라인이 필요한 이유
# 머신러닝 과정은 보통 아래처럼 순서가 필요해요:
# 데이터 전처리 → 모델 학습 → 하이퍼파라미터 튜닝 → 예측
# ✔️ 이걸 파이프라인 없이 하면 각 단계가 따로따로 흩어지게 돼요.
# ✔️ 그런데 파이프라인을 사용하면 한 줄로 전체 흐름을 자동화할 수 있어요.

# ✅ 파이프라인의 장점
# 장점	설명
# 코드 자동화	전처리, 모델링, 튜닝을 하나로 묶음
# 실수 방지	데이터 누락, 스케일링 오류 방지
# 튜닝 연동	GridSearchCV, Cross-Validation과 자연스럽게 통합
# 유지보수	나중에 수정이 매우 쉬움

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 준비
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 파이프라인 정의
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
