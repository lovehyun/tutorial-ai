from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# text = ["get free bitcoin today"]
# vectorizer = CountVectorizer(ngram_range=(1, 2))
# X = vectorizer.fit_transform(text)
# print(vectorizer.get_feature_names_out())
#
# => ['bitcoin' 'bitcoin today' 'free' 'free bitcoin' 'get' 'get free' 'today']


# 3. 모델 정의
model = MultinomialNB()

# 4. 교차 검증 (Stratified K-Fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, labels, cv=skf, scoring='accuracy')

print("[교차검증 정확도 목록]", scores)
print("평균 정확도:", np.mean(scores))

# 교차검증 예측 결과 얻기 (의미 없음)
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import classification_report

# y_pred = cross_val_predict(model, X, labels, cv=skf)
# print(classification_report(labels, y_pred, zero_division=0))


# 5. 전체 데이터로 재학습 (실제 사용을 위해)
model.fit(X, labels)

# 6. 모델 및 벡터 저장
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n모델과 벡터라이저가 저장되었습니다: spam_model.pkl, vectorizer.pkl")
