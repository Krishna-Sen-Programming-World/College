from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

#%% Generate the dataset and display

X, y = make_classification(
    n_features=4,
    n_classes=3,
    n_samples=1000,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,)

plt.scatter(X[:, 0], X[:, 1], c=y, marker="*");
plt.scatter(X[:, 2], X[:, 3], c=y, marker="*");

#%% Create the train-test split and fit the model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=125
)

model = GaussianNB()

# Model training
model.fit(X_train, y_train)

#%% Evaluate the performance of the model

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)

print("Accuracy:", accuray)

labels = [0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();

#%% # Predict Output

print(model.predict([[-1.5, -1.5, 1.5, 1.5]]))
print(model.predict([[-1.5, -1.5, 2, 0]]))
print(model.predict([[3, 3, 3, 3]]))
