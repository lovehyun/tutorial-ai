from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np
import joblib

# 1. 샘플 데이터
texts = [
    "Free money now!!!", "Hi, how are you?", "Claim your prize", "Let's meet tomorrow",
    "Congratulations, you've won!", "Call me when you can", "Click here to get rich",
    "Are you coming to the party?", "Earn cash fast", "See you later",
    "Win a brand new car!", "Lunch at noon?", "You have been selected for a reward",
    "Dinner plans?", "Act now to receive your bonus", "Meeting rescheduled to Friday",
    "Urgent: verify your account", "Thanks for the help", "Get free bitcoin today",
    "Let's go hiking this weekend", "This is not a scam, click now", "Call me back ASAP",
    "Double your income easily", "Join us for a study session", "Exclusive deal just for you"
]

labels = [
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1
]

# 2. 벡터화 (2-gram 포함, 불용어 제거)
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(texts)

# 3. 모델 정의
model = MultinomialNB()

# 3-2. GridSearchCV로 alpha 최적화
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv)
grid.fit(X, labels)

# 4. 최적 모델로 평가
best_model = grid.best_estimator_
y_pred = best_model.predict(X)

print("최적 alpha:", grid.best_params_['alpha'])
print("교차검증 평균 정확도:", grid.best_score_)
# print("\n[전체 데이터 예측 평가]")
# print(classification_report(labels, y_pred, zero_division=0))

# 6. 모델 및 벡터 저장
joblib.dump(best_model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n모델과 벡터라이저가 저장되었습니다: spam_model.pkl, vectorizer.pkl")


# 교차검증 평균 정확도: 0.64
#
# [전체 데이터 예측 평가]
#               precision    recall  f1-score   support
#
#            0       1.00      0.92      0.96        12
#            1       0.93      1.00      0.96        13
#
#     accuracy                           0.96        25
#    macro avg       0.96      0.96      0.96        25
# weighted avg       0.96      0.96      0.96        25

# | 항목                   | 의미                                 | 특징                                    |
# | -------------------- | ---------------------------------- | ------------------------------------- |
# | **교차검증 정확도 (0.64)**  | 데이터를 훈련/검증으로 나눠서 **검증 데이터**에 대해 평가 | **일반화 성능 평가**에 적합 (신뢰도 ↑)             |
# | **전체 데이터 평가 (0.96)** | 전체 데이터를 학습시킨 후 **그 데이터를 그대로 예측**   | **훈련 데이터에 대한 성능**이라 과대평가됨 (과적합 가능성 ↑) |

# 교차검증(0.64)은 더 정직한 평가
# - StratifiedKFold는 데이터를 5등분한 뒤,
#   각 fold에서 80%로 학습, 20%로 검증을 반복합니다.
# - 이때는 모델이 본 적 없는 데이터로 평가하므로 실제 모델의 성능을 더 정확히 반영합니다.
# - 데이터셋이 작고, 학습/검증 분할이 모델에 큰 영향을 줘서 정확도가 낮게 나올 수 있습니다.

# 전체 데이터 평가(0.96)는 "자기 자신을 시험 본 결과"
# - model.fit(X, labels) → 전체 데이터로 모델 학습
# - model.predict(X) → 같은 데이터로 다시 예측
# - → 이건 시험 문제를 외운 학생이 같은 문제로 다시 시험을 본 것과 같아요.
# - 실제 새 데이터에선 이렇게 잘 맞지 않을 가능성이 큽니다.
