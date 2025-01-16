from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#%% Load and explore the dataset and create the train-test split

cancer = datasets.load_breast_cancer()

print("Features:\n", cancer.feature_names)
print("\n\nLabels:\n", cancer.target_names)

print("\n\nData Shape:", cancer.data.shape)
print("Data:\n", cancer.data[0:5])
print("Labels:\n", cancer.target[0:5])

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25,random_state=109)

#%% Import the model, fit it into the training data, run the model on the test data and print the confusion matrix

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print("The Confusion Matrix\n-----------------------")
print(cm)

#%% Print the important evaluation metrics of the model

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1_Score:", metrics.f1_score(y_test, y_pred))