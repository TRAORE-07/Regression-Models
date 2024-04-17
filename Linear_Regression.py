import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\Datasets\\Salary_dataset.csv")

x = data['YearsExperience']
y = data['Salary']

# Sum of the input variables
SumX = np.sum(x)
# Sum of the output variables
SumY = np.sum(y)
# Sum of x square
X2Sum = np.sum(x*x)
# Sum of x by y
SumXY = np.sum(x*y)
# Sum of x at power 2
SumX2 = np.sum(x)**2
# Number of input variables
n = len(x)

a = ((SumY*X2Sum)-(SumX*SumXY))/((n*X2Sum)-SumX2)
b = (n*SumXY - (SumX*SumY))/((n*X2Sum)-SumX2)
# Linear Regression Formula
yp = a + b*x

plt.title("Linear Regression")
plt.scatter(x, y)
plt.plot(x, yp, color='red')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()