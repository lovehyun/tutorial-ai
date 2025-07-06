# uvicorn fastapi_server:app --reload

from fastapi import FastAPI

app = FastAPI()

@app.get("/api/greet/{name}")
async def greet(name: str):
    return {"message": f"안녕하세요, {name}님!"}
