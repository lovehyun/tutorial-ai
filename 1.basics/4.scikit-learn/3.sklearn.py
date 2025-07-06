from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import numpy as np

# ========================
# 1. 데이터셋 불러오기
# ========================
iris = load_iris()
X = iris.data
y = iris.target

print("데이터셋 샘플 (첫 5개):")
print(X[:5])

print("\n타겟 샘플 (첫 5개):")
print(y[:5])

# ========================
# 2. 학습/테스트 데이터 분할
# ========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n학습 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)

# ========================
# 3. 데이터 표준화 (StandardScaler)
# ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("\n표준화된 데이터 샘플:")
print(X_train_scaled[:5])

# ========================
# 4. 라벨 인코딩 (LabelEncoder)
# ========================
labels = ['apple', 'banana', 'apple', 'orange']
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

print("\n라벨 인코딩 결과:")
print("원본 라벨:", labels)
print("인코딩된 값:", encoded_labels)

# ========================
# 5. 결측값 처리 (SimpleImputer)
# ========================
# 결측값 포함 데이터 생성
data_with_nan = np.array([[7, 2], [np.nan, 4], [5, np.nan], [8, 6]])

print("\n결측값 포함 데이터:")
print(data_with_nan)

# 평균으로 결측값 채우기
imputer = SimpleImputer(strategy='mean')
filled_data = imputer.fit_transform(data_with_nan)

print("\n결측값 평균으로 채운 데이터:")
print(filled_data)

# ========================
# 6. 다항 특성 생성 (PolynomialFeatures)
# ========================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train_scaled)

print("\n다항 특성 생성 (2차 다항식):")
print(X_poly[:3])

# ========================
# 7. 특성 선택 (SelectKBest)
# ========================
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

# 상위 5개 특성 선택
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_wine, y_wine)

print("\n원본 wine 데이터 크기:", X_wine.shape)
print("선택된 상위 5개 특성 데이터 크기:", X_selected.shape)

# ========================
# 8. 전처리 파이프라인 구축
# ========================
# 결측값 채우기 + 표준화 + 다항 특성 생성 통합
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# 파이프라인 적용
X_processed = pipeline.fit_transform(data_with_nan)

print("\n파이프라인 적용 결과:")
print(X_processed)

# ========================
# 요약
# ========================
print("\n[총정리]")
print("- 데이터셋 불러오기 (load_iris, load_wine)")
print("- 데이터 분할 (train_test_split)")
print("- 데이터 표준화 (StandardScaler)")
print("- 라벨 인코딩 (LabelEncoder)")
print("- 결측값 처리 (SimpleImputer)")
print("- 다항 특성 생성 (PolynomialFeatures)")
print("- 특성 선택 (SelectKBest)")
print("- 데이터 파이프라인 구축 (Pipeline)")
