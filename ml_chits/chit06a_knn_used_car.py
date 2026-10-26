# Chit 6 (Set A): K-Nearest Neighbors (KNN) Classification using Used Car Price Dataset
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Convert price text like "₹ 4.45 Lakh", "₹ 1.15 Crore", "₹ 60,000" into Lakh
def price_to_lakh(price):
    value = price.replace("₹", "").replace(",", "").strip()
    if "Crore" in value:
        return float(value.replace("Crore", "")) * 100
    if "Lakh" in value:
        return float(value.replace("Lakh", ""))
    return float(value) / 100000


# Load dataset
df = pd.read_csv("used_car_dataset.csv")

# Display first 5 records
print("Dataset:")
print(df.head())

# Convert text columns into numbers
df["Price_Lakh"] = df["car_price_in_rupees"].apply(price_to_lakh)
df["Kms_Driven"] = df["kms_driven"].str.replace(",", "").str.replace("km", "").astype(float)

# Convert price into 3 classes: Low, Medium, High
df["Price_Class"] = pd.qcut(df["Price_Lakh"], q=3, labels=["Low", "Medium", "High"])
print("\nPrice Classes:")
print(df["Price_Class"].value_counts())

# Brand = first word of car name
df["Brand"] = df["car_name"].str.split().str[0]

# Select input features and convert categorical features
X = df[["year_of_manufacture", "Kms_Driven", "Brand", "fuel_type"]]
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Select output
y = df["Price_Class"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (KNN uses distance, so all features must be on the same scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
print(result.head(15))

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

# New car data (2019 Hyundai, Petrol, 35,000 km)
new_car = pd.DataFrame([{"year_of_manufacture": 2019, "Kms_Driven": 35000,
                         "Brand": "Hyundai", "fuel_type": "Petrol"}])
new_car = pd.get_dummies(new_car, dtype=int).reindex(columns=X.columns, fill_value=0)

# Predict new car
prediction = model.predict(scaler.transform(new_car))
print("\nPrediction for New Car:")
print(prediction[0])
