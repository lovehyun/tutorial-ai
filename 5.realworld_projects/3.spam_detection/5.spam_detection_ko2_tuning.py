from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np

# 한글 예제 데이터 (더 많이 추가)
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

# Cross-Validation을 위한 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("="*60)
print("Cross-Validation을 통한 모델 성능 비교")
print("="*60)

# 1. 기본 CountVectorizer + Naive Bayes
print("\n1. CountVectorizer + MultinomialNB")
pipe_count_nb = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

cv_scores = cross_val_score(pipe_count_nb, texts, labels, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 2. TF-IDF + Naive Bayes
print("\n2. TfidfVectorizer + MultinomialNB")
pipe_tfidf_nb = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

cv_scores = cross_val_score(pipe_tfidf_nb, texts, labels, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 3. 문자 단위 n-gram + Naive Bayes
print("\n3. Char n-gram + MultinomialNB")
pipe_char_nb = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 4))),
    ('classifier', MultinomialNB())
])

cv_scores = cross_val_score(pipe_char_nb, texts, labels, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 4. TF-IDF + SVM
print("\n4. TfidfVectorizer + SVM")
pipe_tfidf_svm = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', SVC(probability=True, random_state=42))
])

cv_scores = cross_val_score(pipe_tfidf_svm, texts, labels, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

print("\n" + "="*60)
print("GridSearchCV를 통한 하이퍼파라미터 최적화")
print("="*60)

# GridSearchCV로 최적 파라미터 찾기
param_grid = {
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],  # 단어 조합 범위
    'vectorizer__max_features': [500, 1000, 2000],  # 최대 특성 수
    'vectorizer__min_df': [1, 2],  # 최소 문서 빈도
    'classifier__alpha': [0.1, 0.5, 1.0, 2.0]  # Naive Bayes 스무딩 파라미터
}

# TF-IDF + MultinomialNB 조합으로 최적화
pipe_optimized = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

grid_search = GridSearchCV(
    pipe_optimized, 
    param_grid, 
    cv=cv, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\n최적 파라미터 탐색 중...")
grid_search.fit(texts, labels)

print(f"\n최고 CV 점수: {grid_search.best_score_:.3f}")
print(f"최적 파라미터: {grid_search.best_params_}")

# 최적화된 모델로 최종 평가
print("\n" + "="*60)
print("최적화된 모델 성능 평가")
print("="*60)

best_model = grid_search.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, stratify=labels, random_state=42)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['일반', '스팸'], zero_division=0))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],")
print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")

# 특성 중요도 분석 (TF-IDF 가중치 기준)
print("\n" + "="*60)
print("스팸 판별에 중요한 특성들")
print("="*60)

vectorizer = best_model.named_steps['vectorizer']
classifier = best_model.named_steps['classifier']

# 각 클래스별 로그 확률 차이가 큰 단어들 찾기
feature_names = vectorizer.get_feature_names_out()
log_prob_diff = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]  # 스팸 - 일반
top_spam_indices = log_prob_diff.argsort()[-10:][::-1]  # 상위 10개

print("\n스팸을 나타내는 주요 특성:")
for i, idx in enumerate(top_spam_indices):
    print(f"{i+1:2d}. {feature_names[idx]} (점수: {log_prob_diff[idx]:.3f})")

print("\n" + "="*60)
print("새로운 문장 예측 테스트")
print("="*60)

test_sentences = [
    "축하드립니다! 1억원에 당첨되셨어요!",
    "오늘 저녁에 같이 식사할까요?",
    "지금 바로 가입하면 돈을 드려요",
    "프로젝트 진행 상황은 어떤가요?",
    "긴급! 계정 확인 필요합니다",
    "날씨가 좋네요. 산책할까요?"
]

print("\n최적화된 모델 예측 결과:")
for sentence in test_sentences:
    prediction = best_model.predict([sentence])[0]
    probability = best_model.predict_proba([sentence])[0]
    print(f"문장: {sentence}")
    print(f"예측: {'스팸' if prediction == 1 else '일반'} (스팸 확률: {probability[1]:.3f})")
    print("-" * 50)

# 앙상블 모델 시도
print("\n" + "="*60)
print("앙상블 모델 (추가 성능 향상 시도)")
print("="*60)

from sklearn.ensemble import VotingClassifier

# 여러 모델을 조합한 앙상블
ensemble_model = VotingClassifier([
    ('nb_tfidf', Pipeline([
        ('vectorizer', TfidfVectorizer(**{k.replace('vectorizer__', ''): v 
                                        for k, v in grid_search.best_params_.items() 
                                        if k.startswith('vectorizer')})),
        ('classifier', MultinomialNB(**{k.replace('classifier__', ''): v 
                                       for k, v in grid_search.best_params_.items() 
                                       if k.startswith('classifier')}))
    ])),
    ('svm', Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', SVC(probability=True, random_state=42))
    ])),
    ('char_nb', Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 4))),
        ('classifier', MultinomialNB())
    ]))
], voting='soft')

# 앙상블 모델 CV 평가
cv_scores_ensemble = cross_val_score(ensemble_model, texts, labels, cv=cv, scoring='accuracy')
print(f"Ensemble CV Accuracy: {cv_scores_ensemble.mean():.3f} (+/- {cv_scores_ensemble.std() * 2:.3f})")

print(f"\n성능 개선 요약:")
print(f"- Cross-Validation으로 안정적인 성능 평가")
print(f"- GridSearchCV로 최적 하이퍼파라미터 탐색")
print(f"- 앙상블 모델로 추가 성능 향상 시도")
print(f"- 특성 중요도 분석으로 모델 해석성 제공")
