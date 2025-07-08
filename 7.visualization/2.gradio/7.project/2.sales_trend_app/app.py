# app.py
# pip install tabulate

import gradio as gr
from analysis import load_csv, plot_sales, analyze_trend

with gr.Blocks() as app:
    gr.Markdown("## 월별 판매량 시각화 및 추세 분석")

    with gr.Row():
        # 좌측: 파일 업로드 + 상태 메시지 (세로로 배치)
        with gr.Column():
            csv_input = gr.File(label="CSV 파일 업로드", file_types=[".csv"])
            load_msg = gr.Textbox(label="상태 메시지", lines=2, interactive=False)

        # 우측: CSV 미리보기
        preview_df = gr.Dataframe(
            label="CSV 미리보기",
            row_count=10,      # 최대 10줄
            col_count=(2, "dynamic"),   # (최소 열 수, 동적 or 고정)
            datatype=["str", "number"], # 각 열 타입
            wrap=True,         # 줄 넘김 허용
            interactive=False  # 사용자가 수정하지 못하도록
        )

    load_button = gr.Button("파일 로드")
    # load_button.click(fn=load_csv, inputs=[csv_input], outputs=[load_output])
    load_button.click(fn=load_csv, inputs=[csv_input], outputs=[load_msg, preview_df])

    gr.Markdown("### 판매량 시각화")
    graph_output = gr.Image(label="시각화 그래프", type="pil")
    plot_button = gr.Button("그래프 보기")
    plot_button.click(fn=plot_sales, outputs=[graph_output])

    gr.Markdown("### 추세 분석 및 예측")
    trend_image = gr.Image(label="추세선 포함 그래프", type="pil")
    trend_result = gr.Textbox(label="예측 결과")
    trend_button = gr.Button("추세 분석")
    trend_button.click(fn=analyze_trend, outputs=[trend_image, trend_result])

app.launch()
