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

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# Step 1: Generate synthetic dataset
X, y = make_blobs(n_samples=10000, centers=3, cluster_std=1.0, random_state=42)

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Class'] = y  # Assign cluster labels as the target variable

# Display the first few rows of the DataFrame
print(df.head())
df.info()
print(df.describe())

# Step 2: Prepare features and target variable
X = df.iloc[:, :-1]  # Features (all columns except 'Class')
y = df['Class']       # Target variable ('Class')

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Step 4: Train the Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# Step 5: Plot the Decision Tree with adjustments for clarity
plt.figure(figsize=(20, 12))  # Increase figure size for better visibility
_ = tree.plot_tree(dt,
                   feature_names=X.columns.tolist(),
                   class_names=['Class 0', 'Class 1', 'Class 2'],
                   filled=True,
                   rounded=True,
                   fontsize=12)  # Adjust font size for better readability

plt.title('Decision Tree Visualization')
plt.show()

# Step 6: Make predictions and evaluate the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")

# Step 8: Visualize the Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

