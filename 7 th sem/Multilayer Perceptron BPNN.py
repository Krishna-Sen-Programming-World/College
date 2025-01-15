## Code 4 Multilayer Perceptron BPNN
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df=pd.read_csv('I:\data.csv')
print(df.head())

print(df.shape)  # No parentheses
print(df.describe())

X = df.drop('Class', axis=1)
y = df['Class']

print(X.head())
print(y.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
print(X_train.shape)
print(X_test.shape)
# Create an MLPClassifier model
mlp = MLPClassifier(max_iter=500, activation='relu')
#mlp = MLPClassifier(max_iter=500, activation='tanh')
mlp.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp.predict(X_test)
print(y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
'''
Class,t,y,special_event
series_1,1,5.056,0
series_1,2,5.559,0
series_2,9,9.395,0
series_2,10,9.349,0
series_3,65,17.739,0
series_3,66,19.01,1
series_4,21,19.644,0
series_4,22,19.745,0
'''