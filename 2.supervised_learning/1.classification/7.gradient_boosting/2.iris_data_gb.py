import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Gradient Boosting 모델 생성 및 학습
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# 4. 예측
y_pred = gb.predict(X_test)

# 5. 평가
print("Gradient Boosting 정확도:", accuracy_score(y_test, y_pred))
print("\nGradient Boosting 분류 리포트:\n", classification_report(y_test, y_pred))

# 6. 혼동 행렬 시각화
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
