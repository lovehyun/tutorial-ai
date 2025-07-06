import streamlit as st

st.title("챗봇 페이지")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("질문을 입력하세요:")

if st.button("전송"):
    st.session_state.chat_history.append((user_input, f"답변: {user_input}"))

for i, (q, a) in enumerate(st.session_state.chat_history, 1):
    st.write(f"{i}. {q} ➜ {a}")
