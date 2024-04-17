import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

x = np.linspace(-10, 10, 100)
z = 1/(1+np.exp(-x))

plt.plot(x, z, color='purple')
plt.xlabel("x")
plt.ylabel("sigmoid(x)")