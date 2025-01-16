import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris

#%% Load the data

#Load the data
iris = load_iris()

#Convert it into a dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#Add the target column
df['target'] = iris.target

# Map targets to target names
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}

#Convert the 'target' column from 0, 1, 2 to actual class labels
df['target'] = df['target'].map(target_names)

print(df.head())
print(df.describe())

#%% Prepare the data

#Extracting Independent and dependent Variable  
x = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values

#Do the train-Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.4, random_state=0)

#Do the feature scaling
st_x = StandardScaler() 
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

print("Training Data")
print(x_train)
print("Test Data")
print(x_test)

#%% Fit the model

#classifier= DecisionTreeClassifier(criterion='entropy', random_state=0) 
classifier = DecisionTreeClassifier(class_weight = None, 
                       criterion = 'entropy',
                       max_depth = None,
                       max_features = None,
                       max_leaf_nodes = None,
                       min_samples_leaf = 1,
                       min_samples_split = 2,
                       min_impurity_decrease = 0,
                       random_state = 0) 
classifier.fit(x_train, y_train)

#%% Evaluate the performance of this DT on Test Data

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n\nAccuracy:{:,.2f}%".format(accuracy_score(y_test, y_pred)*100))

#%% Visualize the Decision Tree

text_representation = tree.export_text(classifier)
print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier,
                   feature_names=iris.feature_names,
                   class_names=list(iris.target_names),
                   filled=True)
fig.savefig("decistion_tree.png")