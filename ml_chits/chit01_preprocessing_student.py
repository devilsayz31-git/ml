# Chit 1: Study and Implementation of Data Pre-processing Techniques using Student Dataset
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Step 1: Load dataset
df = pd.read_csv("Student.csv")
print(df)

# Step 2: Check missing values
print(df.isnull())

# Step 3: Count missing values
print(df.isnull().sum())

# Method 1: Remove rows with missing values
df1 = df.dropna()
print(df1)

# Method 2: Fill missing Age with Mean
mean_age = df["Age"].mean()
print("Mean Age:", mean_age)
df["Age"] = df["Age"].fillna(mean_age)

# Method 3: Fill missing Marks with Median
median_marks = df["Marks"].median()
print("Median Marks:", median_marks)
df["Marks"] = df["Marks"].fillna(median_marks)

# Method 4: Fill missing City with Mode
mode_city = df["City"].mode()
print("Mode City:", mode_city[0])
df["City"] = df["City"].fillna(mode_city[0])

# Method 5: Fill missing Attendance and Income using SimpleImputer
imputer = SimpleImputer(strategy="mean")
new_data = imputer.fit_transform(df[["Attendance", "Income"]])
df[["Attendance", "Income"]] = pd.DataFrame(new_data, columns=["Attendance", "Income"])
print(df)

# Final check
print(df.isnull().sum())
