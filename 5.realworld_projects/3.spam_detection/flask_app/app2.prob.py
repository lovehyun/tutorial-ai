from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# 모델과 벡터 불러오기
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    prob = None
    if request.method == 'POST':
        input_text = request.form.get('message')
        if input_text:
            X_input = vectorizer.transform([input_text])
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]  # [정상확률, 스팸확률]

            spam_prob = proba[1]  # 스팸 확률
            result = "스팸" if pred == 1 else "정상"
            prob = f"{spam_prob * 100:.1f}%"

    return render_template('index2.html', result=result, prob=prob)

if __name__ == '__main__':
    app.run(debug=True)
