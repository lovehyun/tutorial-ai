import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 산점도
plt.figure()
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=iris_df)
plt.title('Sepal Length vs Width')
plt.show()

# 산점도 (꽃잎)
plt.figure()
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='target', data=iris_df)
plt.title('Petal Length vs Width')
plt.show()

# 히스토그램
plt.figure()
sns.histplot(iris_df['sepal length (cm)'], kde=True, bins=20)
plt.title('Sepal Length Distribution')
plt.show()

# 박스플롯
plt.figure()
sns.boxplot(x='target', y='petal length (cm)', data=iris_df)
plt.title('Petal Length by Class')
plt.show()

# 바이올린 플롯
plt.figure()
sns.violinplot(x='target', y='petal width (cm)', data=iris_df)
plt.title('Petal Width by Class (Violin Plot)')
plt.show()

# 페어플롯
sns.pairplot(iris_df, hue='target')
plt.show()
