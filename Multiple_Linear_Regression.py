import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('D:\\Datasets\\CarPrice.csv')

# Extract the relevant columns
x1 = data['carlength'].values.reshape(-1, 1)
x2 = data['enginesize'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

# Combine x1 and x2 into a single matrix for the features
X = np.hstack((x1, x2))

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict values for the plane
x1_range = np.linspace(x1.min(), x1.max(), 10)
x2_range = np.linspace(x2.min(), x2.max(), 10)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
y_grid = model.predict(np.c_[x1_grid.ravel(), x2_grid.ravel()]).reshape(x1_grid.shape)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the original data points
ax.scatter(x1, x2, y, color='red', label='Data points')

# Plot the regression plane
ax.plot_surface(x1_grid, x2_grid, y_grid, color='blue', alpha=0.5, rstride=100, cstride=100)

# Set labels
ax.set_xlabel('Car Length')
ax.set_ylabel('Engine Size')
ax.set_zlabel('Price')
ax.set_title('Multiple Linear Regression')

# Display the plot
plt.show()

# Print the coefficients and intercept
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R^2 Score: {model.score(X, y)}")