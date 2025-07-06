# 지도학습 3단계: 모델 평가 지표 (Accuracy, Precision, Recall, F1-Score)
# ✅ 1. 왜 평가 지표를 배워야 할까?
# 👉 단순히 정확도(Accuracy)만 보는 것은 위험하기 때문이에요.
# 👉 특히 불균형 데이터나 다중 클래스 문제에서는 Precision, Recall, F1-Score가 훨씬 중요합니다.

# ✅ 2. 주요 평가 지표 정리
# 평가 지표	의미	핵심 포인트
# Accuracy (정확도)	맞춘 비율	전체 중 맞춘 비율 (가장 기본)
# Precision (정밀도)	True로 예측한 것 중 실제 True	False Positive를 얼마나 줄였는가
# Recall (재현율)	실제 True 중 맞춘 비율	False Negative를 얼마나 줄였는가
# F1-Score	Precision과 Recall의 조화 평균	Precision, Recall 균형 평가

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. 데이터 로드 및 분할
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. 예측
y_pred = model.predict(X_test)

# 4. 평가 지표 계산
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n=== 상세 리포트 ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
