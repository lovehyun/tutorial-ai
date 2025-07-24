from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 한글 예제 데이터
# texts = [
#     "무료로 돈을 받으세요!!!",
#     "안녕하세요, 어떻게 지내세요?",
#     "축하합니다! 당첨되셨습니다",
#     "내일 만나요",
#     "지금 바로 클릭하세요! 대박 이벤트",
#     "언제 시간 있을 때 연락주세요",
#     "돈 벌기 쉬운 방법 알려드려요",
#     "파티에 오실 건가요?",
#     "빠른 현금 수익! 지금 가입",
#     "나중에 봐요",
#     "긴급! 계정이 정지됩니다",
#     "오늘 날씨가 좋네요",
#     "100% 확실한 투자 기회",
#     "회의는 몇 시에 시작하나요?",
#     "믿을 수 없는 할인 혜택!",
#     "주말 잘 보내세요"
# ]

# # 레이블 (1: 스팸, 0: 일반)
# labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

texts = [
    "무료로 돈을 받으세요!!!",
    "안녕하세요, 어떻게 지내세요?",
    "축하합니다! 당첨되셨습니다",
    "내일 만나요",
    "지금 바로 클릭하세요! 대박 이벤트",
    "언제 시간 있을 때 연락주세요",
    "돈 벌기 쉬운 방법 알려드려요",
    "파티에 오실 건가요?",
    "빠른 현금 수익! 지금 가입",
    "나중에 봐요",
    "긴급! 계정이 정지됩니다",
    "오늘 날씨가 좋네요",
    "100% 확실한 투자 기회",
    "회의는 몇 시에 시작하나요?",
    "믿을 수 없는 할인 혜택!",
    "주말 잘 보내세요",
    "무료 체험 지금 신청하세요!",
    "커피 한 잔 어때요?",
    "대출 승인! 바로 받아보세요",
    "생일 축하해요!",
    "한정 특가! 놓치면 후회",
    "점심 뭐 먹을까요?",
    "즉시 당첨! 1000만원",
    "일찍 들어가세요",
    "광고) 최저가 보장",
    "가족들과 잘 지내세요",
    "수익률 300% 보장합니다",
    "내일 비 온다던데",
    "무료 상담 받아보세요",
    "잘 자요"
]

# 레이블 (1: 스팸, 0: 일반)
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

print("=== 기본 CountVectorizer (공백 기준 토큰화) ===")
# 1. 기본 벡터화 (공백 기준으로 단어 분리)
vectorizer_basic = CountVectorizer()
X_basic = vectorizer_basic.fit_transform(texts)

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X_basic, labels, test_size=0.3, random_state=42, stratify=labels)

# 모델 학습
model_basic = MultinomialNB()
model_basic.fit(X_train, y_train)

# 예측 및 평가
y_pred = model_basic.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n[단어 사전 (일부)]")
vocab = vectorizer_basic.vocabulary_
print({k: v for k, v in sorted(vocab.items())[:10]})


print("\n" + "="*50)
print("=== 문자 단위 n-gram (char-level) ===")
# 2. 문자 단위 벡터화 (한글에 더 적합할 수 있음)
vectorizer_char = CountVectorizer(
    analyzer='char',  # 문자 단위
    ngram_range=(2, 3),  # 2-3글자 조합
    max_features=1000  # 상위 1000개 특성만 사용
)
X_char = vectorizer_char.fit_transform(texts)

# 학습/테스트 분리
X_train_char, X_test_char, y_train_char, y_test_char = train_test_split(
    X_char, labels, test_size=0.3, stratify=labels, random_state=42
)

# 모델 학습
model_char = MultinomialNB()
model_char.fit(X_train_char, y_train_char)

# 예측 및 평가
y_pred_char = model_char.predict(X_test_char)
print("Accuracy:", accuracy_score(y_test_char, y_pred_char))
print("Classification Report:")
print(classification_report(y_test_char, y_pred_char, zero_division=0))

print("\n[문자 n-gram 특성 (일부)]")
char_vocab = vectorizer_char.vocabulary_
print({k: v for k, v in sorted(char_vocab.items())[:10]})


print("\n" + "="*50)
print("=== 새로운 문장 예측 테스트 ===")
test_sentences = [
    "축하드립니다! 1억원에 당첨되셨어요!",
    "오늘 저녁에 같이 식사할까요?",
    "지금 바로 가입하면 돈을 드려요",
    "프로젝트 진행 상황은 어떤가요?"
]


print("\n[기본 벡터화 결과]")
for sentence in test_sentences:
    X_new = vectorizer_basic.transform([sentence])
    prediction = model_basic.predict(X_new)[0]
    probability = model_basic.predict_proba(X_new)[0]
    print(f"문장: {sentence}")
    print(f"예측: {'스팸' if prediction == 1 else '일반'} (스팸 확률: {probability[1]:.3f})")
    print()

print("\n[문자 n-gram 결과]")
for sentence in test_sentences:
    X_new_char = vectorizer_char.transform([sentence])
    prediction_char = model_char.predict(X_new_char)[0]
    probability_char = model_char.predict_proba(X_new_char)[0]
    print(f"문장: {sentence}")
    print(f"예측: {'스팸' if prediction_char == 1 else '일반'} (스팸 확률: {probability_char[1]:.3f})")
    print()
