# 지도학습 5단계: 불균형 데이터 처리
# ✔️ 왜 필요한가?
# 불균형 데이터에서는 대부분의 평가 지표가 편향된 결과를 보여줄 수 있습니다.

# 특히 다수 클래스(정상 데이터)를 기준으로만 학습하면, 소수 클래스(이상 데이터)를 제대로 분류하지 못합니다.

# ✅ 주요 기법 2가지
# 방법	설명
# Class Weight	소수 클래스의 중요도를 높여주는 방법
# Sampling	데이터 개수를 강제로 맞추는 방법 (Over/Under Sampling)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. 데이터 로드 및 분할
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 클래스 가중치 조정
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Class Weight 적용 결과 ===")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
