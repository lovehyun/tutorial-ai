import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
texts = [
    "Free money now!!!", "Hi, how are you?", "Claim your prize", "Let's meet tomorrow",
    "Congratulations, you've won!", "Call me when you can", "Click here to get rich",
    "Are you coming to the party?", "Earn cash fast", "See you later"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# 벡터화 및 모델 학습
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 데이터 분리 (성능 확인용)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=labels)

# 학습
model = MultinomialNB()
model.fit(X_train, y_train)
# model.fit(X, labels) # 전체 다를 학습

# 성능 평가
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 저장 (성능이 괜찮은 경우만)
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
