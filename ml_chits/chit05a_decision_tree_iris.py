# Chit 5 (Set A): Decision Tree Classification using IRIS Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# STEP 1: LOAD DATASET
data = pd.read_csv("Iris.csv")
data.columns = data.columns.str.strip()
print("First 5 Records:")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nMissing Values:")
print(data.isnull().sum())

# STEP 2: SEPARATE INPUT AND OUTPUT
X = data.drop("Species", axis=1)

# STEP 3: REMOVE ID COLUMN
X = X.drop("Id", axis=1)

# STEP 4: ENCODE TARGET VARIABLE
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Species"])
print("\nTarget Classes:")
print(label_encoder.classes_)

# STEP 5: SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 6: CREATE DECISION TREE MODEL
model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)

# STEP 7: TRAIN MODEL
model.fit(X_train, y_train)

# STEP 8: PREDICTION
y_pred = model.predict(X_test)

# STEP 9: ACCURACY
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100, "%")

# STEP 10: CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# STEP 11: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# STEP 12: CONFUSION MATRIX GRAPH
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Iris Classification")
plt.colorbar()
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()

# STEP 13: VISUALIZE DECISION TREE
plt.figure(figsize=(25, 12))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(x) for x in label_encoder.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
