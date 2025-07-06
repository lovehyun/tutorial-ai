# 1_python_basics/1_numpy_intro.py
import numpy as np

print("NumPy 배열 생성")

# 1차원 배열
arr1 = np.array([1, 2, 3, 4, 5])
print(f"1차원 배열: {arr1}")

# 2차원 배열
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2차원 배열:\n{arr2}")

print("\nNumPy 기본 연산")
print(f"배열 덧셈: {arr1 + 5}")
print(f"배열 제곱: {arr1 ** 2}")
print(f"배열 평균: {np.mean(arr1)}")
print(f"배열 합계: {np.sum(arr1)}")

print("\n배열 인덱싱/슬라이싱")
print(f"첫 번째 원소: {arr1[0]}")
print(f"2~4번째 원소: {arr1[1:4]}")
