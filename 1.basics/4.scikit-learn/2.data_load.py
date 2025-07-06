# 데이터셋	    샘플 수	    특성 개수	    타겟
# iris	        150	        4	        꽃 종류 3개
# wine	        178	        13	        와인 종류 3개
# digits	    1797	    64	        0~9 숫자
# breast_cancer	569	        30	        양성/음성

from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
import pandas as pd

# 1. iris 데이터셋 불러오기
iris = load_iris()

print("=== iris 데이터셋 구조 ===")
print("키 목록:", iris.keys())

print("\n특성 데이터 샘플:")
print(iris.data[:5])

print("\n타겟 데이터 샘플:")
print(iris.target[:5])

print("\n타겟 이름:")
print(iris.target_names)

print("\n특성 이름:")
print(iris.feature_names)

# pandas DataFrame으로 변환
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print("\niris DataFrame 샘플:")
print(iris_df.head())

print("\niris DataFrame 통계 요약:")
print(iris_df.describe())

print("\niris 타겟 분포:")
print(iris_df['target'].value_counts())

# =========================================

# 2. wine 데이터셋 불러오기
wine = load_wine()

print("\n=== wine 데이터셋 구조 ===")
print("키 목록:", wine.keys())

print("\n특성 데이터 샘플:")
print(wine.data[:5])

print("\n타겟 이름:")
print(wine.target_names)

wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

print("\nwine DataFrame 샘플:")
print(wine_df.head())

print("\nwine 타겟 분포:")
print(wine_df['target'].value_counts())

# =========================================

# 3. digits 데이터셋 불러오기
digits = load_digits()

print("\n=== digits 데이터셋 구조 ===")
print("키 목록:", digits.keys())

print("\n특성 데이터 샘플 (8x8 이미지):")
print(digits.data[0].reshape(8, 8))

print("\n타겟 데이터 샘플:")
print(digits.target[:10])

digits_df = pd.DataFrame(digits.data)
digits_df['target'] = digits.target

print("\ndigits DataFrame 샘플:")
print(digits_df.head())

print("\ndigits 타겟 분포:")
print(digits_df['target'].value_counts())

# =========================================

# 4. breast_cancer 데이터셋 불러오기
cancer = load_breast_cancer()

print("\n=== breast_cancer 데이터셋 구조 ===")
print("키 목록:", cancer.keys())

print("\n특성 데이터 샘플:")
print(cancer.data[:5])

print("\n타겟 이름:", cancer.target_names)

cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target

print("\nbreast_cancer DataFrame 샘플:")
print(cancer_df.head())

print("\nbreast_cancer 타겟 분포:")
print(cancer_df['target'].value_counts())
