import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = pd.read_csv('D:\\Datasets\\CarPrice.csv')

x1 = data['carlength']
x2 = data['enginesize']
y = data['price'].values.reshape(-1, 1)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1, x2, y)
plt.show()