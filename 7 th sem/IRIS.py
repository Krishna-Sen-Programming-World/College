
### Question 1(IRIS data)
''' 
 Load IRIS data from the SKLearn library

ure:.

2. List down the feature names and their range of values [output-1]

3. List down the class names and the number of instances in each class [output-2]

4. Normalize the data

5. Split the data in 60-40 proportion

6. Use the

a. K Nearest Neighbour-classifier to classify the test data with K = 1

b. Fit a Gaussian Naïve Bayes classifier model on the training data and classify the test data

C. Fit a Decision Tree classifier model on the training data and classify the test data

d. Fit a MLP classifier (one hidden layer with 8 nodes) model on the training data and classify the test data

7. Print the confusion matrix [output-3]

'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load IRIS data
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
feature_names = iris.feature_names
class_names = iris.target_names

# Step 2: List feature names and their range of values
print("Feature Names and Ranges [Output-1]:")
for i, feature in enumerate(feature_names):
    feature_min = X[:, i].min()
    feature_max = X[:, i].max()
    print(f"{feature}: Min = {feature_min}, Max = {feature_max}")

# Step 3: List class names and number of instances
print("\nClass Names and Instance Counts [Output-2]:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {np.sum(y == i)} instances")

# Step 4: Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Step 5: Split the data (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.4, random_state=42)

# Step 6a: K Nearest Neighbour Classifier (K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# Step 6b: Gaussian Naïve Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)

# Step 6c: Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
dtc_predictions = dtc.predict(X_test)

# Step 6d: MLP Classifier (One hidden layer with 8 nodes)
mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)

# Step 7: Print confusion matrices
print("\nConfusion Matrices [Output-3]:")
print("\nKNN Classifier:")
print(confusion_matrix(y_test, knn_predictions))
print("\nGaussian Naïve Bayes Classifier:")
print(confusion_matrix(y_test, gnb_predictions))
print("\nDecision Tree Classifier:")
print(confusion_matrix(y_test, dtc_predictions))
print("\nMLP Classifier:")
print(confusion_matrix(y_test, mlp_predictions))