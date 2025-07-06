import streamlit as st

st.title("여러 세션 상태 관리")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if 'text' not in st.session_state:
    st.session_state.text = ""

if st.button("카운터 증가"):
    st.session_state.counter += 1

new_text = st.text_input("내용 입력", value=st.session_state.text)

if st.button("내용 저장"):
    st.session_state.text = new_text

st.write(f"카운터: {st.session_state.counter}")
st.write(f"저장된 내용: {st.session_state.text}")
