import gradio as gr

def add_one(count):
    return count + 1, f"현재 값: {count + 1}"

with gr.Blocks() as demo:
    gr.Markdown("# State 사용 예제")

    counter = gr.State(value=0)  # 초기값 0

    count_button = gr.Button("1 증가하기")
    output = gr.Textbox(label="현재 값")

    count_button.click(add_one, inputs=counter, outputs=[counter, output])

demo.launch()
