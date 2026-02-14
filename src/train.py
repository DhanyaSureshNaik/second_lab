import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
model_dir = os.path.join(os.path.dirname(__file__), "../model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "cancer_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")
