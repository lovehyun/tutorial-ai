from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("model_lstm_multiclass_seq10.h5")
data = np.load("lstm_data.npz")
X = data["X"]
y = data["y"].astype(int)

y_cat = to_categorical(y, num_classes=3)
_, X_test, _, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
