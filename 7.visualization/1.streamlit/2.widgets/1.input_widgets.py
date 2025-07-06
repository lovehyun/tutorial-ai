import streamlit as st

st.title("Streamlit 입력 위젯 실습")

# 텍스트 입력
name = st.text_input("이름을 입력하세요:")

# 숫자 입력
age = st.number_input("나이를 입력하세요:", min_value=0, max_value=120, step=1)

# 날짜 선택
birth_date = st.date_input("생일을 선택하세요:")

# 라디오 버튼
gender = st.radio("성별을 선택하세요:", ("남성", "여성", "기타"))

if st.button("결과 보기"):
    st.write(f"이름: {name}")
    st.write(f"나이: {age}")
    st.write(f"생일: {birth_date}")
    st.write(f"성별: {gender}")
