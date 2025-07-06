# 지도학습 7단계: Precision-Recall Curve 최적 Threshold 찾기
# ✔️ 핵심 개념
# Precision과 Recall의 적절한 균형을 자동으로 계산해서 최적 Threshold를 찾는 과정입니다.

# 일반적으로 F1-Score가 가장 높은 지점이 최적의 임계값입니다.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 분할
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 모델 학습
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 3. 확률 예측
y_proba = model.predict_proba(X_test)[:, 1]

# 4. Precision-Recall Curve 생성
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# 5. F1-Score 계산
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"최적 Threshold: {best_threshold:.4f}")
print(f"최고 F1-Score: {f1_scores[best_index]:.4f}")

# 6. 최적 Threshold 적용
y_pred_adjusted = (y_proba >= best_threshold).astype(int)

print("\n=== 최적 Threshold 적용 결과 ===")
print(classification_report(y_test, y_pred_adjusted, target_names=cancer.target_names))

# 7. 시각화
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.axvline(x=best_threshold, color='red', linestyle='--', label='Best Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid()
plt.show()


# 핵심 설명
# 항목	설명
# F1-Score 계산	Precision, Recall의 조화 평균
# 최적 Threshold	F1-Score가 가장 높은 임계값 자동 추출
# 시각화	최적 Threshold 위치를 그래프로 표시

# ✔️ 이렇게 하면 임계값을 객관적으로 최적화할 수 있습니다.
# ✔️ 단순히 0.5 기준으로 자르지 않고, 실제 모델에 맞는 최적 기준을 찾을 수 있어요.
