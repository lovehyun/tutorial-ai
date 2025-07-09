# pip install imbalanced-learn

# SMOTE (Synthetic Minority Over-sampling Technique)는
# 소수 클래스(공격 데이터)를 가짜 샘플로 증가시켜서 학습 데이터의 균형을 맞춰주는 방법입니다.
# 항목	내용
# 방식	소수 클래스 데이터를 기반으로 근처 점을 보간해 가짜 데이터를 생성
# 결과	데이터가 더 균형 있게 되어, 모델이 소수 클래스도 잘 학습함
#
# 불균형 전:
# - 정상: 950개  
# - DoS: 30개  
# - Probe: 20개
# SMOTE 적용 후:
# - 정상: 950개  
# - DoS: 950개 (SMOTE로 증강)  
# - Probe: 950개 (SMOTE로 증강)
# => 데이터 수를 맞춰줌으로써 학습에 공정한 기회를 부여합니다.

# 3_model_training_smote.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter

# 데이터 로드
df = pd.read_csv("network_multiclass.csv")

# 일부만 선택하여 불균형 시뮬레이션
normal = df[df['label'] == 0].sample(n=600, random_state=42)
dos = df[df['label'] == 1].sample(n=80, random_state=42)
probe = df[df['label'] == 2].sample(n=40, random_state=42)
imbalanced_df = pd.concat([normal, dos, probe]).sample(frac=1, random_state=42).reset_index(drop=True)

X = imbalanced_df.drop(columns='label')
y = imbalanced_df['label']

print("클래스 분포 (SMOTE 전):", Counter(y))

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE 적용
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print("클래스 분포 (SMOTE 후):", Counter(y_resampled))

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# 예측
y_pred = model.predict(X_test_scaled)

# 평가
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
