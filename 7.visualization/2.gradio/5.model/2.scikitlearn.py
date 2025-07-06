# Scikit-learn 모델 연동
# 결과: 입력한 꽃잎/꽃받침 크기로 아이리스 품종을 예측합니다.
# pip install scikit-learn

import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 데이터 및 모델 준비
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction[0]]

app = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs=gr.Textbox(label="예측 결과")
)

app.launch()
