import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\Datasets\\disease.csv")

# Separate features and target variable
x = data.iloc[:, 0:15]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Initialize and train the Decision Tree classifier
DT = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
DT.fit(x_train, y_train)

# Predict on the test set
y_pred = DT.predict(x_test)

# Plot the decision tree
plt.figure(figsize=(11, 10))
tree.plot_tree(DT, filled=True, feature_names=data.columns[:-1], class_names=True)
plt.show()

# Calculate and print the accuracy
score = DT.score(x_test,y_test)
print("Predict Accuracy: ", score, "%")