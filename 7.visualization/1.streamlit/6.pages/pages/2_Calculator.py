import streamlit as st

st.title("계산기 페이지")

num1 = st.number_input("숫자 1")
num2 = st.number_input("숫자 2")
operation = st.selectbox("연산 선택", ["더하기", "빼기", "곱하기", "나누기"])

if st.button("계산하기"):
    if operation == "더하기":
        result = num1 + num2
    elif operation == "빼기":
        result = num1 - num2
    elif operation == "곱하기":
        result = num1 * num2
    elif operation == "나누기":
        result = num1 / num2 if num2 != 0 else "0으로 나눌 수 없습니다."
    st.write(f"결과: {result}")
