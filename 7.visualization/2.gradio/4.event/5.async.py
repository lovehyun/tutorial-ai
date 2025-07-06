import gradio as gr
import asyncio

async def long_task(name):
    await asyncio.sleep(3)  # 3초 대기 (비동기)
    return f"처리 완료: {name}님!"

with gr.Blocks() as demo:
    name_input = gr.Textbox(label="이름 입력")
    btn = gr.Button("비동기 처리")
    output = gr.Textbox()

    btn.click(fn=long_task, inputs=name_input, outputs=output)

demo.launch()
