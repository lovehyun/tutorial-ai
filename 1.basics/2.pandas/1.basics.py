import pandas as pd

# 딕셔너리 → 데이터프레임
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
}
df = pd.DataFrame(data)

print("데이터프레임 미리보기:")
print(df)

# 통계 요약
print("\n평균 점수:", df['Score'].mean())
print("최대 나이:", df['Age'].max())
