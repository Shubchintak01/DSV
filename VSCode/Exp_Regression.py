# 1. Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

Y = np.array([
    30000, 35000, 40000, 45000, 50000,
    55000, 60000, 65000, 70000, 75000
])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Prediction
Y_pred = model.predict(X)

# Display coefficient and intercept
print("Simple Linear Regression")
print("------------------------")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict salary for 11 years of experience
experience = np.array([[11]])
predicted_salary = model.predict(experience)

print("Predicted Salary for 11 years:",
      predicted_salary[0])

# Calculate accuracy measures
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")

plt.legend()
plt.show()

# 2. Multiple Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3500],

    "Bedrooms": [2, 2, 3, 3, 3,
                 4, 4, 4, 5, 5],

    "Age": [10, 8, 5, 7, 4,
            3, 6, 2, 5, 1],

    "Price": [40, 45, 55, 60, 68,
              75, 82, 90, 98, 110]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Independent variables
X = df[["Area", "Bedrooms", "Age"]]

# Dependent variable
Y = df["Price"]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, Y_train)

# Prediction
Y_pred = model.predict(X_test)

# Display coefficients
print("\nMultiple Linear Regression")
print("--------------------------")

print("Intercept:", model.intercept_)

print("Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

# Display actual and predicted values
result = pd.DataFrame({
    "Actual Price": Y_test.values,
    "Predicted Price": Y_pred
})

print("\nActual vs Predicted:")
print(result)

# Calculate evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Predict price for a new house
new_house = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Age": [5]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price of New House:",
      predicted_price[0], "Lakhs")

