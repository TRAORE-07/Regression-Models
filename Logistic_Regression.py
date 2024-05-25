import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Plot the sigmoid function
x = np.linspace(-10, 10, 100)
z = 1 / (1 + np.exp(-x))

plt.title("Sigmoid Function")
plt.plot(x, z, color='purple')
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.show()

# Load the dataset
data = pd.read_csv('D:\\Datasets\\Social_Network_Ads.csv')

# Preprocess the data (this is an example, adjust as necessary)
# Assuming the last column is the target and others are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

