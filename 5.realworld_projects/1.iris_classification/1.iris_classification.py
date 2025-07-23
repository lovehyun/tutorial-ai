from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# 1. 데이터 로드
X, y = load_iris(return_X_y=True)

# 2. 데이터 분할 (훈련용 / 테스트용)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # y값을 균등하게 포함
)

# 3. 정규화 (스케일링)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 생성 및 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 5. 모델 & 스케일러 저장
joblib.dump(model, "lr_model.pkl")
joblib.dump(scaler, "scaler.pkl")  # 추론 시에도 필요함!

# 6. 예측 및 평가
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
