import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#%% Import the dataset

# load the diabetes dataset
pima = pd.read_csv("D:\Sukumar\TIU Materials\Machine Learning\Odd Semester 2024\Lab Experiments\Assignment-7\diabetes.csv")
pima.head()

#%% do the train test splitting and scale the data
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = pima[feature_cols] # Features
y = pima.Outcome #Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% create the model, do the prediction and print the confusion matrix

logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

# do the prediction
y_pred = logreg.predict(X_test)

# Calculate and print the confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)