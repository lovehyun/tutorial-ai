import streamlit as st

st.title("버튼 클릭 카운터")

if 'count' not in st.session_state:
    st.session_state.count = 0

if st.button("클릭"):
    st.session_state.count += 1

st.write(f"버튼이 {st.session_state.count}번 클릭되었습니다.")
