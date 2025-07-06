import streamlit as st
import httpx
import asyncio

st.title("⚡ FastAPI 연동 (비동기 요청)")

name = st.text_input("이름을 입력하세요:")

async def call_api(name):
    url = f"http://localhost:8000/api/greet/{name}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

if st.button("FastAPI 호출"):
    if name:
        result = asyncio.run(call_api(name))
        st.write(result['message'])
