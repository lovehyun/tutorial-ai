# Hugging Face 사전학습 모델 연동
# 결과: 입력한 문장의 감정을 실시간으로 분석해 줍니다.
# pip install transformers

import gradio as gr
from transformers import pipeline

# 사전 학습된 감정 분석 모델 불러오기
classifier = pipeline("sentiment-analysis")

def sentiment_analysis(text):
    result = classifier(text)[0]
    return f"라벨: {result['label']}, 확신도: {result['score']:.2f}"

app = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(label="문장을 입력하세요"),
    outputs=gr.Textbox(label="감정 분석 결과")
)

app.launch()
