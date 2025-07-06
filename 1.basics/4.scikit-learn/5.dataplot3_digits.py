import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 데이터 로드
digits = load_digits()

# 샘플 이미지 출력
plt.figure(figsize=(10, 4))
for index, (image, label) in enumerate(zip(digits.images[:10], digits.target[:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
plt.suptitle('Digits Sample Images')
plt.show()

# DataFrame 생성
digits_df = pd.DataFrame(digits.data)
digits_df['target'] = digits.target

# 타겟 분포 카운트플롯
plt.figure()
sns.countplot(x='target', data=digits_df)
plt.title('Digits Class Frequency')
plt.show()

# 산점도 (픽셀 0 vs 픽셀 1)
plt.figure()
sns.scatterplot(x=0, y=1, hue='target', data=digits_df, palette='tab10')
plt.title('Pixel 0 vs Pixel 1 by Digit')
plt.show()

# 상관계수 히트맵 (픽셀 상관 관계)
plt.figure(figsize=(12, 10))
sns.heatmap(digits_df.corr(), cmap='coolwarm', cbar=False)
plt.title('Digits Feature Correlation')
plt.show()

# 페어플롯 (픽셀 0~2)
subset = digits_df[[0, 1, 2, 'target']]
sns.pairplot(subset, hue='target', palette='tab10')
plt.show()
