import numpy as np

# 배열 생성
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 벡터 연산
print("a + b:", a + b)
print("a * b:", a * b)
print("Dot product:", np.dot(a, b))

# 2차원 배열 생성
matrix = np.array([[1, 2], [3, 4]])
print("Matrix:\n", matrix)
print("Transpose:\n", matrix.T)
