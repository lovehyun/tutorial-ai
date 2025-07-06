import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 선 그래프 (lineplot)
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
sns.lineplot(x=x, y=y)
plt.title('Seaborn Line Plot')
plt.xlabel('x')
plt.ylabel('y = sin(x)')
plt.grid()
plt.show()

# 2. 산점도 (scatterplot)
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)

plt.figure()
sns.scatterplot(x=x, y=y)
plt.title('Seaborn Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# 3. 히스토그램 (histplot)
data = np.random.randn(1000)

plt.figure()
sns.histplot(data, bins=30, kde=True)
plt.title('Seaborn Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 4. 박스플롯 (boxplot)
data_box = pd.DataFrame({
    '과목': ['수학', '수학', '영어', '영어', '과학', '과학'] * 10,
    '점수': np.random.randint(60, 100, 60)
})

plt.figure()
sns.boxplot(x='과목', y='점수', data=data_box)
plt.title('Seaborn Box Plot')
plt.xlabel('과목')
plt.ylabel('점수')
plt.show()

# 5. 카운트플롯 (countplot)
data_count = pd.DataFrame({
    '카테고리': np.random.choice(['A', 'B', 'C'], 100)
})

plt.figure()
sns.countplot(x='카테고리', data=data_count)
plt.title('Seaborn Count Plot')
plt.xlabel('카테고리')
plt.ylabel('빈도')
plt.show()

# 6. 히트맵 (heatmap)
corr = np.random.rand(5, 5)

plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Seaborn Heatmap')
plt.show()
