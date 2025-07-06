import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# 데이터 로드
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

# 박스플롯
plt.figure()
sns.boxplot(x='target', y='alcohol', data=wine_df)
plt.title('Alcohol by Wine Class')
plt.show()

# 바이올린 플롯
plt.figure()
sns.violinplot(x='target', y='malic_acid', data=wine_df)
plt.title('Malic Acid by Wine Class')
plt.show()

# 산점도
plt.figure()
sns.scatterplot(x='alcohol', y='malic_acid', hue='target', data=wine_df)
plt.title('Alcohol vs Malic Acid')
plt.show()

# 히스토그램
plt.figure()
sns.histplot(wine_df['color_intensity'], bins=20, kde=True)
plt.title('Color Intensity Distribution')
plt.show()

# 카운트플롯
plt.figure()
sns.countplot(x='target', data=wine_df)
plt.title('Wine Class Count')
plt.show()

# 상관계수 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(wine_df.corr(), cmap='coolwarm', annot=False)
plt.title('Wine Feature Correlation')
plt.show()

# 페어플롯 (상위 4개 특성)
top4 = wine_df.corr()['target'].abs().sort_values(ascending=False).index[1:5]
sns.pairplot(wine_df[top4.to_list() + ['target']], hue='target')
plt.show()
