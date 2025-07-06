# 지도학습 6단계: Threshold 튜닝 (임계값 조정)
# ✔️ 왜 Threshold 튜닝이 필요할까?
# 대부분의 분류 모델은 확률로 예측을 반환합니다.
# 👉 기본적으로 0.5를 기준으로 0이면 음성, 1이면 양성으로 분류해요.
# ✔️ 하지만 불균형 데이터에서는 0.5 기준이 적절하지 않은 경우가 많아요.
# ✔️ 임계값을 조절하면 더 많은 양성 (소수 클래스) 탐지가 가능합니다.

# ✅ 핵심 목표
# 모델이 예측한 확률을 기준으로 Precision / Recall 균형을 최적화하는 Threshold를 찾는 것.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

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

# 5. PR 커브 시각화
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid()
plt.show()

# 6. 임계값 설정 및 평가
optimal_threshold = 0.4  # 예시: 직접 조정 가능
y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)

print(f"\n=== Threshold: {optimal_threshold} 적용 결과 ===")
print(classification_report(y_test, y_pred_adjusted, target_names=cancer.target_names))


#  핵심 포인트
# 항목	설명
# predict_proba	확률로 예측 결과 반환
# precision_recall_curve	임계값 변화에 따른 Precision / Recall 계산
# Threshold 튜닝	0.5가 아닌 최적의 임계값을 직접 찾는 과정

# ✔️ 임계값을 낮추면 → 더 많은 True Positive를 탐지 (Recall 증가, Precision 감소)
# ✔️ 임계값을 올리면 → 더 정확하게 Positive를 탐지 (Precision 증가, Recall 감소)
