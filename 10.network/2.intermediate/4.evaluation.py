# 4_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer

# 데이터 로드
df = pd.read_csv("network_multiclass.csv")
X = df.drop(columns="label")
y = df["label"]

# 다중 클래스용 레이블 바이너리화 (ROC용)
y_bin = label_binarize(y, classes=[0, 1, 2])

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y)

# 모델 학습 (Random Forest 사용)
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# 예측
y_pred_bin = model.predict(X_test)
y_pred = y_pred_bin.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Confusion Matrix 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification Report 출력
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(6, 4))
sns.heatmap(report_df.iloc[:3, :-1], annot=True, cmap="YlGnBu")
plt.title("Precision / Recall / F1-score by Class")
plt.show()

# ROC Curve (One-vs-Rest 방식)
fpr = {}
tpr = {}
roc_auc = {}
n_classes = y_test.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
