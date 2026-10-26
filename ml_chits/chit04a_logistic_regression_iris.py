# Chit 4 (Set A): Implementation and Evaluation of Logistic Regression Model using IRIS Dataset
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv("Iris.csv")
print("========== DATASET ==========")
print(df.head())

# 3. Separate Features and Target
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n========== TRAINING DATA ==========")
print(X_train.head())
print("\n========== TESTING DATA ==========")
print(X_test.head())

# 5. Create Logistic Regression Model
model = LogisticRegression(max_iter=200)

# 6. Train Model
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# 7. Model Parameters
print("\n========== MODEL PARAMETERS ==========")
print("Coefficients:")
for cls, coefs in zip(model.classes_, model.coef_):
    print(cls, dict(zip(X.columns, coefs.round(4))))
print("\nIntercept:", model.intercept_)

# 8. Predictions
y_pred = model.predict(X_test)
print("\n========== PREDICTIONS ==========")
print("Actual values:")
print(y_test.values)
print("\nPredicted values:")
print(y_pred)

# 9. Probabilities
probabilities = model.predict_proba(X_test)
print("\n========== PROBABILITIES ==========")
print(probabilities.round(4))

# 10. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n========== ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n========== CONFUSION MATRIX ==========")
print(cm)

# 12. Classification Report
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# 13. Predict New Flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
prediction = model.predict(new_flower)
probability = model.predict_proba(new_flower)
print("\n========== NEW FLOWER ==========")
print("Prediction:", prediction[0])
print("Probabilities:", probability[0].round(4))
