##Code 3 SVM
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('I:\data.csv')
print(df.head())

X = df.iloc[:,1:]
y = df.iloc[:, 0] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


clf1=SVC(kernel='linear')
clf1.fit(X_train, y_train.values.ravel())
# Make predictions
y_pred = clf1.predict(X_test)
print(y_pred)
# Evaluate the model
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy:.2f}")
# Radius Basis Funcrion RBF Kernel SVM
clf2=SVC(kernel='rbf')
clf2.fit(X_train, y_train.values.ravel())
# Make predictions
y_pred = clf2.predict(X_test)
print(y_pred)
# Evaluate the model
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy:.2f}")


## Polynomial Kernel SVM
clf3 = SVC(kernel='poly')  
clf3.fit(X_train, y_train.values.ravel())
# Make predictions
y_pred = clf3.predict(X_test)
print(y_pred)
# Evaluate the model
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy:.2f}")


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Weight', y='Length1', hue='Class', alpha=0.7)
plt.title('Weight vs Length1 by Class')
plt.show()
#or
fig=px.scatter_3d(df, x='Weight', y='Length1', z='Length2', color ='Class')
fig.show()



'''
Species,Weight,Length1,Length2,Length3,Height,Class
Bream,430,26.5,29,34,12.444,5.134
Roach,78,17.5,18.8,21.2,5.5756,2.9044
Parkki,55,13.5,14.7,16.5,6.8475,2.3265
Whitefish,540,28.5,31,34,10.744,6.562
'''