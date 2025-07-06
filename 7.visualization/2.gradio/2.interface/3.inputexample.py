import gradio as gr

def greet(name):
    return f"안녕하세요, {name}님!"

app = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="이름을 입력하세요"),
    outputs=gr.Textbox(label="인사 메시지"),
    examples=["철수", "영희", "민수"]
)

app.launch()
