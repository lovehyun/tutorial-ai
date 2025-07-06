import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import uvicorn

# FastAPI 메인 서버
app = FastAPI()

@app.get("/")
async def home():
    return {"message": "FastAPI 메인 서버"}

# Gradio 앱
def greet(name):
    return f"안녕하세요, {name}님!"

gr_app = gr.Interface(fn=greet, inputs="text", outputs="text")

# FastAPI에 Gradio mount (FastAPI만 지원)
# Flask는 WSGI 기반, FastAPI는 ASGI 기반이라 Gradio가 FastAPI용으로만 내부 통합을 지원하는 구조입니다.
app = mount_gradio_app(app, gr_app, path="/gradio")

# 실행: uvicorn fastapi_gradio_combined:app --reload
# uvicorn은 파일명에서 숫자로 시작하는 모듈을 인식하지 못합니다.
if __name__ == "__main__":
    # uvicorn을 코드 내에서 직접 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
