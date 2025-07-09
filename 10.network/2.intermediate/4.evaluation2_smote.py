# 4_evaluation_smote.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

# 데이터 로드 및 불균형 시뮬레이션
df = pd.read_csv("network_multiclass.csv")
normal = df[df['label'] == 0].sample(n=600, random_state=42)
dos = df[df['label'] == 1].sample(n=80, random_state=42)
probe = df[df['label'] == 2].sample(n=40, random_state=42)
imbalanced_df = pd.concat([normal, dos, probe]).sample(frac=1, random_state=42).reset_index(drop=True)

X = imbalanced_df.drop(columns='label')
y = imbalanced_df['label']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------- 모델1: SMOTE 적용 안 함 ---------
model_no_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_no_smote.fit(X_train_scaled, y_train)
pred_no_smote = model_no_smote.predict(X_test_scaled)

# --------- 모델2: SMOTE 적용 ---------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_resampled, y_resampled)
pred_smote = model_smote.predict(X_test_scaled)

# --------- 평가 함수 ---------
def evaluate_model(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(report_df.iloc[:3, :-1], annot=True, cmap="YlGnBu")
    plt.title(f"{title} - Precision / Recall / F1")
    plt.show()

# --------- 평가 결과 출력 ---------
print("모델1 - SMOTE 미적용")
evaluate_model(y_test, pred_no_smote, "No SMOTE")

print("모델2 - SMOTE 적용")
evaluate_model(y_test, pred_smote, "SMOTE")
