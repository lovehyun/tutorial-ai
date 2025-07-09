# 4_evaluation_roc.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE

# 데이터 로드 및 불균형 시뮬레이션
df = pd.read_csv("network_multiclass.csv")
normal = df[df['label'] == 0].sample(n=600, random_state=42)
dos = df[df['label'] == 1].sample(n=80, random_state=42)
probe = df[df['label'] == 2].sample(n=40, random_state=42)
imbalanced_df = pd.concat([normal, dos, probe]).sample(frac=1, random_state=42).reset_index(drop=True)

X = imbalanced_df.drop(columns='label')
y = imbalanced_df['label']
y_bin = label_binarize(y, classes=[0, 1, 2])

# 학습/테스트 분할
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, stratify=y, test_size=0.2, random_state=42)
y_train = y_train_bin.argmax(axis=1)
y_test = y_test_bin.argmax(axis=1)

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE 적용 전 모델 학습
model_no_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_no_smote.fit(X_train_scaled, y_train)
probs_no_smote = model_no_smote.predict_proba(X_test_scaled)

# SMOTE 적용 후 모델 학습
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_resampled, y_resampled)
probs_smote = model_smote.predict_proba(X_test_scaled)

# ROC Curve 함수
def plot_multiclass_roc(y_test_bin, probs, title):
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = y_test_bin.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# 시각화
print("ROC Curve - SMOTE 미적용")
plot_multiclass_roc(y_test_bin, probs_no_smote, "ROC Curve (No SMOTE)")

print("ROC Curve - SMOTE 적용")
plot_multiclass_roc(y_test_bin, probs_smote, "ROC Curve (SMOTE Applied)")
