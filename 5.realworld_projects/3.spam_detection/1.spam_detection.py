from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 예제용 텍스트 데이터
texts = [
    "Free money now!!!",
    "Hi, how are you?",
    "Claim your prize",
    "Let's meet tomorrow",
    "Congratulations, you've won!",
    "Call me when you can",
    "Click here to get rich",
    "Are you coming to the party?",
    "Earn cash fast",
    "See you later"
]

# 각 문장에 대한 레이블 (1: spam, 0: ham)
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# 1. 백터화 (문장을 자동으로 단어로 나눠줌)
vectorizer = CountVectorizer() # 내부적으로 소문자로 변환, 문장 부호 제거, 토큰화 등 진행해서 백터 형태로 표현함
X = vectorizer.fit_transform(texts)

# 2. 학습/테스트 분리
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=3, stratify=labels)
# stratify=labels를 추가하면 train/test 모두에 스팸과 햄이 골고루 들어가도록 조정됩니다.

# 3. 모델 학습 (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n[Classification Report]")
# print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior. _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
# -> 모델이 스팸(1) 클래스를 단 한 개도 예측하지 못했다는 뜻입니다.
# zero_division=0을 추가하면 0으로 처리하되 경고는 발생하지 않음.

# 5. 단어 사전 출력 (선택)
print("\n[단어 인덱스]")
print(vectorizer.vocabulary_)


# Accuracy: 0.8
#
# [Classification Report]
#               precision    recall  f1-score   support
#
#            0       1.00      0.67      0.80         3
#            1       0.67      1.00      0.80         2
#
#     accuracy                           0.80         5
#    macro avg       0.83      0.83      0.80         5
# weighted avg       0.87      0.80      0.80         5

# | 항목        | 0 (일반메일, ham)           | 1 (스팸메일, spam)                |
# | --------- | ----------------------- | ----------------------------- |
# | precision | 1.00 (정확히 ham 예측함)      | 0.67 (스팸이라고 한 것 중 67%만 진짜 스팸) |
# | recall    | 0.67 (진짜 ham 중 67%만 맞춤) | 1.00 (진짜 스팸을 모두 맞춤)           |
# | f1-score  | 0.80                    | 0.80                          |
# | support   | 3                       | 2                             |

# 의미 있는 관찰 포인트
#  - 모델이 스팸(1) 은 전부 잘 맞췄고 (recall=1.0),
#  - 일반메일(0) 은 1개 정도를 실수로 스팸이라고 잘못 예측했을 가능성이 있습니다.
#  - precision이 높지만 recall이 낮은 경우: 모델이 보수적으로 예측하고 있다는 뜻 (실수로 스팸을 ham으로 보진 않지만, ham을 스팸으로 오판할 수도 있음).
