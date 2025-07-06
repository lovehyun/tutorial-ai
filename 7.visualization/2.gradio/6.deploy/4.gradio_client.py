# gradio_client_async.py
import gradio as gr
import httpx
import asyncio

# 비동기 FastAPI 요청
async def call_fastapi(name):
    url = "http://localhost:8000/api/greet"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"name": name})
        if response.status_code == 200:
            return response.json().get("message", "응답 없음")
        else:
            return "서버 에러"

# Gradio 앱 (비동기 지원)
app = gr.Interface(
    fn=call_fastapi,
    inputs=gr.Textbox(label="이름을 입력하세요"),
    outputs=gr.Textbox(label="FastAPI 응답 결과"),
    live=True
)

app.launch()
