# 1_numpy_basics/1_numpy_matrix_operations.py
import numpy as np

# 1. 2차원 배열(행렬) 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("행렬 A:")
print(A)

print("\n행렬 B:")
print(B)

# 2. 행렬 덧셈
C = A + B
print("\n행렬 덧셈 (A + B):")
print(C)

# 3. 행렬 뺄셈
D = A - B
print("\n행렬 뺄셈 (A - B):")
print(D)

# 4. 원소별 곱셈 (Hadamard Product)
E = A * B
print("\n원소별 곱셈 (A * B):")
print(E)

# 5. 행렬 곱셈 (Matrix Product)
F = np.dot(A, B)
print("\n행렬 곱셈 (A @ B 또는 np.dot(A, B)):")
print(F)

# 6. 전치 행렬 (Transpose)
A_T = A.T
print("\nA의 전치 행렬:")
print(A_T)

# 7. 역행렬 (Inverse)
A_inv = np.linalg.inv(A)
print("\nA의 역행렬:")
print(A_inv)

# 8. 단위 행렬 (Identity Matrix)
I = np.eye(2)
print("\n2x2 단위 행렬:")
print(I)

# 9. 행렬식 (Determinant)
det_A = np.linalg.det(A)
print("\nA의 행렬식:")
print(det_A)

# 10. 고유값과 고유벡터
eigvals, eigvecs = np.linalg.eig(A)
print("\nA의 고유값:")
print(eigvals)
print("\nA의 고유벡터:")
print(eigvecs)

# 11. 브로드캐스팅 (스칼라 + 행렬)
G = A + 10
print("\n브로드캐스팅 (모든 원소에 10 더하기):")
print(G)

# 12. 행렬 축별 연산 (합계)
print("\n행별 합계 (axis=1):")
print(np.sum(A, axis=1))

print("\n열별 합계 (axis=0):")
print(np.sum(A, axis=0))

# 13. 행렬 슬라이싱
print("\n행렬 슬라이싱: 첫 번째 행")
print(A[0, :])  # 첫 번째 행

print("\n행렬 슬라이싱: 첫 번째 열")
print(A[:, 0])  # 첫 번째 열

# 14. 블록 행렬 생성 (np.hstack, np.vstack)
H = np.hstack([A, B])  # 좌우 결합
V = np.vstack([A, B])  # 상하 결합

print("\n블록 행렬 (좌우 결합):")
print(H)

print("\n블록 행렬 (상하 결합):")
print(V)

# 15. 무작위 행렬 생성
np.random.seed(0)  # 재현성 고정
R = np.random.randint(0, 10, (3, 3))
print("\n3x3 무작위 행렬:")
print(R)

# 16. 대각 행렬 추출
print("\n행렬의 대각 원소:")
print(np.diag(A))

# 17. 행렬 차원 확인
print("\nA의 차원:", A.shape)

# 18. 행렬 복사 (깊은 복사 vs 얕은 복사)
original = np.array([[1, 2], [3, 4]])
shallow_copy = original
deep_copy = original.copy()

original[0, 0] = 100

print("\n얕은 복사 (shallow_copy):")
print(shallow_copy)

print("\n깊은 복사 (deep_copy):")
print(deep_copy)
