from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 모델과 스케일러 로드
# model = joblib.load("models/lr_model.pkl")      # LinearRegression 또는 다른 회귀 모델
model = joblib.load("models/gb_model.pkl")      # GradientBoosting 또는 다른 회귀 모델
scaler = joblib.load("models/scaler.pkl")       # StandardScaler

feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                 'Population', 'AveOccup', 'Latitude', 'Longitude']

# 홈 페이지
@app.route('/')
def home():
    return render_template('index.html')

# 예측 처리
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 입력값 받기
        input_values = [float(request.form[f]) for f in feature_names]
        input_df = pd.DataFrame([input_values], columns=feature_names)
                
        # 정규화 후 예측
        scaled_array = scaler.transform(input_df)
        
        # ndarray → 다시 DataFrame으로 변환
        scaled_df = pd.DataFrame(scaled_array, columns=feature_names)

        # 예측 시에도 feature name 포함된 DataFrame 사용
        prediction = model.predict(scaled_df)[0]

        result = round(prediction, 2)
        
        return render_template('index.html', prediction_text=f"예측된 중간 주택 가격: ${result * 100000:,.0f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"오류 발생: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


# Voting Regressor
#  - 여러 회귀 모델(예: LinearRegression, RandomForest, GradientBoosting 등)의 예측값을 평균내어 최종 결과를 산출하는 방식입니다.
#  - 단순 평균 방식의 앙상블.
# Stacking Regressor
#  - 여러 모델의 예측 결과를 **다시 하나의 모델(메타 모델)**에 학습시켜 최종 예측값을 도출하는 방식입니다.
#  - 더 복잡하지만, 일반적으로 Voting보다 더 성능이 좋습니다.
# MSE (Mean Squared Error, 평균 제곱 오차)
#  - 모델의 예측값과 실제값 간의 차이를 제곱해서 평균낸 값입니다.
#  - 값이 작을수록 예측 성능이 좋다는 뜻입니다.
# 
# 결과 해석
# | 모델                 | MSE  | 해석                                              |
# | ------------------ | ---- | ----------------------------------------------- |
# | Voting Regressor   | 0.31 | 여러 모델의 예측 평균을 낸 결과, 평균 제곱 오차가 **0.31**          |
# | Stacking Regressor | 0.25 | 여러 모델의 예측 결과를 학습한 최종 모델이, 더 낮은 오차인 **0.25**를 달성 |
