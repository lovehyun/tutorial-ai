import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 예시: 선 그래프
x = np.linspace(0, 10, 100)
y = np.sin(x)

st.title('Seaborn 그래프 Streamlit 시각화')

fig, ax = plt.subplots()
sns.lineplot(x=x, y=y, ax=ax)
ax.set_title('Seaborn Line Plot')

st.pyplot(fig)
