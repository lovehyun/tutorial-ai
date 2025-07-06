import matplotlib.pyplot as plt
import numpy as np

# 1. 여러 그래프를 한 화면에 (서브플롯)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(12, 5))

# 첫 번째 그래프
plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째
plt.plot(x, y1, color='blue')
plt.title('sin(x) 그래프')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째
plt.plot(x, y2, color='red')
plt.title('cos(x) 그래프')
plt.xlabel('x')
plt.ylabel('cos(x)')
plt.grid()

plt.tight_layout()
plt.show()

# 2. 히스토그램 (데이터 분포 시각화)
data = np.random.randn(1000)  # 평균 0, 표준편차 1인 정규분포 데이터

plt.hist(data, bins=30, color='purple', alpha=0.7)
plt.title('히스토그램: 데이터 분포')
plt.xlabel('값')
plt.ylabel('빈도')
plt.grid()
plt.show()

# 3. 산점도 (Scatter Plot)
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)  # y = 2x + noise

plt.scatter(x, y, color='green', alpha=0.6)
plt.title('산점도: y = 2x + noise')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# 4. 막대 그래프 (Bar Chart)
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.bar(categories, values, color='orange')
plt.title('막대 그래프')
plt.xlabel('카테고리')
plt.ylabel('값')
plt.show()

# 5. 파이 차트 (Pie Chart)
sizes = [30, 20, 25, 25]
labels = ['사과', '바나나', '체리', '포도']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('파이 차트: 과일 비율')
plt.show()

# 6. 커스터마이징 (라인 스타일, 마커, 색상)
plt.plot(x, y, linestyle='--', marker='o', color='black')
plt.title('커스터마이징: 점선 + 원형 마커')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# 7. 범례 (Legend) 추가
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title('범례 예제')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# 8. 그래프 저장하기
plt.plot(x, y1)
plt.title('그래프 저장 예제')
plt.savefig('saved_plot.png')  # 현재 경로에 그래프 저장
plt.close()  # 창 닫기 (안하면 계속 누적됨)

# 9. 투명도 및 색상 조절
plt.scatter(x, y1, alpha=0.3, color='red', label='low alpha')
plt.scatter(x, y2, alpha=0.7, color='blue', label='high alpha')
plt.title('투명도 비교')
plt.legend()
plt.show()
