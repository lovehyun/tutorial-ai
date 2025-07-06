import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("여러 그래프를 가로로 배치하기")

col1, col2, col3 = st.columns(3)

# 그래프 1 (col1)
with col1:
    fig1, ax1 = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    sns.lineplot(x=x, y=y, ax=ax1)
    ax1.set_title('선 그래프')
    st.pyplot(fig1)

# 그래프 2 (col2)
with col2:
    fig2, ax2 = plt.subplots()
    x = np.random.rand(100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    sns.scatterplot(x=x, y=y, ax=ax2)
    ax2.set_title('산점도')
    st.pyplot(fig2)

# 그래프 3 (col3)
with col3:
    fig3, ax3 = plt.subplots()
    data = np.random.randn(1000)
    sns.histplot(data, bins=30, kde=True, ax=ax3)
    ax3.set_title('히스토그램')
    st.pyplot(fig3)
