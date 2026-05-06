# machine_learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Data
X = np.array([[37000], [45000], [52000], [60000]])  # <<< INPUT DATA
y = np.array([6.2, 6.5, 7.1, 7.4])                # <<< INPUT DATA

# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions (train/test)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Errors
train_error = mean_squared_error(y_train, train_pred)
test_error = mean_squared_error(y_test, test_pred)

print("Train Error:", train_error)
print("Test Error:", test_error)

# Get equation
m = model.coef_[0]
b = model.intercept_
print(f"Equation: y = {m}x + {b}")

# Predict NEW values
X_future = np.array([[38000], [50000], [65000]])   # <<< INPUT DATA

future_predictions = model.predict(X_future)

print("Future Predictions:")
for x, y_pred in zip(X_future, future_predictions):
    print(f"GDP: {x[0]} → Predicted Life Satisfaction: {y_pred}")

# Create range (FROM → TO)
X_range = np.linspace(35000, 700000, 48).reshape(-1, 1)   # <<< INPUT DATA

# Predict for full range
y_range_pred = model.predict(X_range)

# Print results
for x, y_pred in zip(X_range, y_range_pred):
    print(f"GDP: {int(x[0])} → Predicted Life Satisfaction: {y_pred:.2f}")
