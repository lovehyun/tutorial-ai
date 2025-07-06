import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("탭으로 그래프 구분하기")

tab1, tab2, tab3 = st.tabs(["선 그래프", "산점도", "히스토그램"])

with tab1:
    fig1, ax1 = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    sns.lineplot(x=x, y=y, ax=ax1)
    ax1.set_title('선 그래프')
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    x = np.random.rand(100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    sns.scatterplot(x=x, y=y, ax=ax2)
    ax2.set_title('산점도')
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    data = np.random.randn(1000)
    sns.histplot(data, bins=30, kde=True, ax=ax3)
    ax3.set_title('히스토그램')
    st.pyplot(fig3)
