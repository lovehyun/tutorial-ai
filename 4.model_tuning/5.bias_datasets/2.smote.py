# SMOTE는 소수 클래스 데이터를 가짜로 생성해서 데이터 개수를 강제로 맞추는 기법입니다.
# imblearn 라이브러리가 필요합니다. (pip install imbalanced-learn)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. 데이터 로드 및 분할
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. SMOTE 준비
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"SMOTE 적용 후 클래스 분포: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

# 3. 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

print("\n=== SMOTE 적용 결과 ===")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
