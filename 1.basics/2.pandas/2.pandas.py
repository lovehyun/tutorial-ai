# 주제	            설명
# 필터링	        조건문으로 원하는 데이터 추출
# 그룹화	        GroupBy 집계, 여러 통계 연산
# 병합	            Merge로 데이터 결합
# 결측값 처리	    평균으로 대체, 결측값 제거
# 피벗 테이블	    테이블 재구성 (엑셀 피벗처럼)
# 정렬	            데이터 내림차순/오름차순 정렬
# 중복 제거	        중복 데이터 삭제
# 시계열 데이터	    날짜 생성, 이동 평균 처리
# 데이터 요약	    info, describe로 데이터 개괄 파악

import pandas as pd
import numpy as np

# 1. 데이터 생성
data = {
    '이름': ['철수', '영희', '민수', '지민', '철수', '영희', '민수', '지민'],
    '과목': ['수학', '수학', '수학', '수학', '영어', '영어', '영어', '영어'],
    '점수': [90, 85, 95, 88, 80, 70, 100, 90]
}

df = pd.DataFrame(data)

print("원본 데이터:")
print(df)

# 2. 데이터 필터링
print("\n수학 점수 90점 이상 학생:")
print(df[(df['과목'] == '수학') & (df['점수'] >= 90)])

# 3. 그룹화 (GroupBy) + 집계 (Aggregation)
print("\n학생별 평균 점수:")
print(df.groupby('이름')['점수'].mean())

print("\n과목별 최고 점수:")
print(df.groupby('과목')['점수'].max())

# 4. 그룹화 + 여러 집계 함수 사용
print("\n학생별 점수 통계 (최소, 최대, 평균):")
print(df.groupby('이름')['점수'].agg(['min', 'max', 'mean']))

# 5. 데이터 병합 (Merge)
students = pd.DataFrame({
    '이름': ['철수', '영희', '민수', '지민'],
    '학년': [1, 2, 3, 1]
})

merged = pd.merge(df, students, on='이름')

print("\n학생 정보 병합 (학년 포함):")
print(merged)

# 6. 결측값 처리
data_with_na = {
    '이름': ['철수', '영희', '민수', '지민'],
    '점수': [np.nan, 85, np.nan, 90]
}

df_na = pd.DataFrame(data_with_na)

print("\n결측값 포함 데이터:")
print(df_na)

# 결측값 평균으로 채우기
df_na['점수'] = df_na['점수'].fillna(df_na['점수'].mean())

print("\n결측값 평균으로 채운 데이터:")
print(df_na)

# 7. 피벗 테이블
pivot = df.pivot_table(index='이름', columns='과목', values='점수', aggfunc='mean')

print("\n피벗 테이블 (학생별 과목 점수):")
print(pivot)

# 8. 데이터 정렬
print("\n점수 기준 내림차순 정렬:")
print(df.sort_values(by='점수', ascending=False))

# 9. 중복 데이터 처리
df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

print("\n중복 데이터 포함:")
print(df_dup)

print("\n중복 제거:")
print(df_dup.drop_duplicates())

# 10. 시계열 데이터 생성 및 처리
dates = pd.date_range(start='2025-01-01', periods=7, freq='D')
sales = pd.Series([100, 150, 200, 250, 300, 350, 400], index=dates)

print("\n시계열 데이터:")
print(sales)

print("\n7일 이동 평균:")
print(sales.rolling(window=3).mean())

# 11. 데이터프레임 정보 요약
print("\n데이터프레임 요약 정보:")
print(df.info())

print("\n데이터프레임 기본 통계 요약:")
