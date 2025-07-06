# 1_python_basics/2_pandas_intro.py
import pandas as pd

print("Pandas Series 생성")
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)

print("\nPandas DataFrame 생성")
data = {
    '이름': ['철수', '영희', '민수'],
    '점수': [85, 90, 95]
}
df = pd.DataFrame(data)
print(df)

print("\n데이터 조회")
print(f"첫 번째 학생: {df.iloc[0]}")
print(f"'점수' 컬럼:\n{df['점수']}")

print("\n데이터 요약")
print(df.describe())
