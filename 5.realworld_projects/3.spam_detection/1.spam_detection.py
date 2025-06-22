from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 간단한 예제용 텍스트 데이터
texts = ["Free money now!!!", "Hi, how are you?", "Claim your prize", "Let's meet tomorrow"]
labels = [1, 0, 1, 0]  # 1: spam, 0: ham

# 전처리
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 학습
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5)
model = MultinomialNB()
model.fit(X_train, y_train)

# 평가
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
