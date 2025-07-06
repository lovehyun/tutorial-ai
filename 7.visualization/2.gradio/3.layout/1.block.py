# Blocks란?
# Blocks는 Gradio에서 레이아웃을 자유롭게 설계할 수 있게 해주는 고급 API입니다.
# Interface보다 더 유연하고 복잡한 앱을 만들 때 사용합니다.
#
# 주요 레이아웃 컴포넌트
# 컴포넌트	설명
# Row	가로 배치
# Column	세로 배치
# Tab	탭 메뉴
# Accordion	펼치기/접기 메뉴
# State	앱 내 변수 저장 (세션 유지)

import gradio as gr

def calc_add(x, y):
    return x + y

def calc_mul(x, y):
    return x * y

with gr.Blocks() as demo:
    gr.Markdown("# 간단한 계산기")

    with gr.Row():
        x_input = gr.Number(label="숫자 X")
        y_input = gr.Number(label="숫자 Y")

    with gr.Row():
        add_button = gr.Button("덧셈")
        mul_button = gr.Button("곱셈")

    result_output = gr.Textbox(label="결과")

    add_button.click(calc_add, inputs=[x_input, y_input], outputs=result_output)
    mul_button.click(calc_mul, inputs=[x_input, y_input], outputs=result_output)

demo.launch()
