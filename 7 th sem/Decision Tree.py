##Code 5 Decision Tree
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df=pd.read_csv('I:\data.csv')
print(df.head())
df.info()
print(df.describe())
X = df.iloc[:, 1:5]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(X_train.shape)
print(X_test.shape)
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

from sklearn import tree
# Function to plot the decision tree
fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(dt, feature_names=X.columns.tolist(), class_names=['L', 'B', 'R'], filled=True)
# or attributes/features and class name= yes,no here l/b/r
#_ = tree.plot_tree(dt, feature_names=['weather','temperature','Humidity','wind'], class_names=['L', 'B', 'R'], filled=True)
plt.show()
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")

import seaborn as sns
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
'''
age,sex,bp,colestrol,heartdisease
R,1,1,1,2
R,1,1,1,3
L,2,3,1,4
L,2,3,2,1
B,1,1,1,1
B,2,3,2,3
'''