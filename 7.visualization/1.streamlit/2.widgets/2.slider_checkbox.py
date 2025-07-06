import streamlit as st

st.title("슬라이더와 체크박스 실습")

st.write("슬라이더를 이용해 값을 선택하세요.")
value = st.slider("값 선택", min_value=0, max_value=100, step=5)

show_double = st.checkbox("값을 두 배로 보기")

if show_double:
    st.write(f"선택한 값의 두 배: {value * 2}")
else:
    st.write(f"선택한 값: {value}")
