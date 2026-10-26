# Chit 2: Implementation and Evaluation of Linear Regression Model using given Dataset
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 2: Create Dataset
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Price': [98, 95, 88, 83, 81, 79, 75, 71, 69, 65, 62, 58, 52, 47, 43, 34, 28, 23, 21, 20]
}
df = pd.DataFrame(data)
print(df)

# Step 3: Independent and Dependent Variables
X = df[['Age']]
Y = df['Price']

# Step 4: Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Step 5: Create Model
model = LinearRegression()

# Step 6: Train Model
model.fit(X_train, Y_train)

# Step 7: Prediction
Y_pred = model.predict(X_test)
print("Predicted Price:")
print(Y_pred)

# Step 8: Model Parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 9: Evaluation
print("MAE :", mean_absolute_error(Y_test, Y_pred))
print("MSE :", mean_squared_error(Y_test, Y_pred))
print("R2 Score :", r2_score(Y_test, Y_pred))

# Step 10: Visualization
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=3, label='Regression Line')
plt.xlabel("Age")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
