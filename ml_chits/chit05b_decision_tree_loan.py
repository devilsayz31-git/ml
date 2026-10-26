# Chit 5 (Set B): Decision Tree Classification using Loan Approval dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# STEP 1: LOAD DATASET
data = pd.read_csv("loan_approval_data.csv")
data.columns = data.columns.str.strip()
print("First 5 Records:")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nColumn Names:")
print(data.columns)
print("\nMissing Values:")
print(data.isnull().sum())

# STEP 2: HANDLE MISSING VALUES
numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
for column in numeric_columns:
    data[column] = data[column].fillna(data[column].median())

categorical_columns = data.select_dtypes(include=["object", "string"]).columns
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# STEP 3: SEPARATE INPUT AND OUTPUT
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# STEP 4: REMOVE APPLICANT ID COLUMN
X = X.drop("Applicant_ID", axis=1)

# STEP 5: CONVERT CATEGORICAL FEATURES
X = pd.get_dummies(X, drop_first=True)

# STEP 6: ENCODE TARGET VARIABLE
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("\nTarget Classes:")
print(label_encoder.classes_)

# STEP 7: SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 8: CREATE DECISION TREE MODEL
model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)

# STEP 9: TRAIN MODEL
model.fit(X_train, y_train)

# STEP 10: PREDICTION
y_pred = model.predict(X_test)

# STEP 11: ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100, "%")

# STEP 12: CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# STEP 13: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# STEP 14: CONFUSION MATRIX GRAPH
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Loan Approval Prediction")
plt.colorbar()
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()

# STEP 15: VISUALIZE DECISION TREE
plt.figure(figsize=(25, 12))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(x) for x in label_encoder.classes_],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.show()
