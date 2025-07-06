import gradio as gr

def file_info(file):
    return f"파일명: {file.name}, 크기: {file.size} bytes"

with gr.Blocks() as demo:
    file_input = gr.File(label="파일 업로드")
    output = gr.Textbox()

    file_input.upload(fn=file_info, inputs=file_input, outputs=output)

demo.launch()
