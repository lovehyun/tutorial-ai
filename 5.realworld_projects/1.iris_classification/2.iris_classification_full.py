# iris_ml_workflow.py

# 1. 데이터 불러오기
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# 2. 데이터 전처리 - 정규화(StandardScaler)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. 데이터 분할 (학습용/테스트용)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 모델 학습
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True)
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()

rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

# 5. 모델 저장
import joblib

joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# 6. 모델 앙상블
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Soft Voting
ensemble = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('svm', svm_model),
    ('knn', knn_model)
], voting='soft')

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
print("Soft Voting Accuracy:", accuracy_score(y_test, ensemble_pred))

# Weighted Voting
weighted_ensemble = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('svm', svm_model),
    ('knn', knn_model)
], voting='soft', weights=[2, 2, 1])

weighted_ensemble.fit(X_train, y_train)
weighted_pred = weighted_ensemble.predict(X_test)
print("Weighted Voting Accuracy:", accuracy_score(y_test, weighted_pred))

# Stacking
stack_model = StackingClassifier(estimators=[
    ('rf', rf_model),
    ('svm', svm_model),
    ('knn', knn_model)
], final_estimator=LogisticRegression(), cv=5)

stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, stack_pred))
