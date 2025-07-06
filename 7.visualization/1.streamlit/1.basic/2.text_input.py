import streamlit as st

st.title("✍️ 텍스트 입력과 버튼")

name = st.text_input("이름을 입력하세요:")

if st.button("인사하기"):
    st.write(f"안녕하세요, {name}님!")
