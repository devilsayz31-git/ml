# Chit 4 (Set B): Implementation and Evaluation of Logistic Regression Model using student dataset
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Create Dataset
data = {
    "Study_Hours": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    "Attendance": [45, 50, 55, 60, 65, 68, 70, 75, 78, 80, 85, 88, 90, 92, 95],
    # 0 = Fail, 1 = Pass
    "Pass": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("========== DATASET ==========")
print(df)

# 3. Separate Features and Target
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n========== TRAINING DATA ==========")
print(X_train)
print("\n========== TESTING DATA ==========")
print(X_test)

# 5. Create Logistic Regression Model
model = LogisticRegression()

# 6. Train Model
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# 7. Model Parameters
print("\n========== MODEL PARAMETERS ==========")
print("Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_[0]):
    print(feature, "=", coefficient)
print("\nIntercept:", model.intercept_[0])

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
print(probabilities)

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
print(classification_report(y_test, y_pred, zero_division=0))

# 13. Predict New Student (Study_Hours = 5, Attendance = 80)
new_student = pd.DataFrame({"Study_Hours": [5], "Attendance": [80]})
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)
print("\n========== NEW STUDENT ==========")
print("Study Hours:", 5)
print("Attendance:", 80)
print("\nPrediction:", prediction[0])
print("Fail Probability:", probability[0][0])
print("Pass Probability:", probability[0][1])
if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")
