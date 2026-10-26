# Chit 6 (Set B): K-Nearest Neighbors (KNN) Classification using Iris Dataset
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("Iris.csv")

# Display first 5 records
print("Dataset:")
print(df.head())

# Select input features
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Select output
y = df["Species"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display actual and predicted values
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print("\nActual vs Predicted:")
print(result)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# New flower data
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
)

# Predict new flower
prediction = model.predict(new_flower)
print("\nPrediction for New Flower:")
print(prediction[0])
