# fastapi_server.py
# uvicorn fastapi_server:app --reload

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 요청 데이터 구조 정의
class NameRequest(BaseModel):
    name: str

@app.post("/api/greet")
async def greet(request: NameRequest):
    name = request.name
    return {"message": f"안녕하세요, {name}님!"}
