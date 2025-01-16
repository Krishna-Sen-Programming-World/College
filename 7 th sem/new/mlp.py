import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#%% Read the data and load it into a data frame

df = pd.read_csv(r"D:\Sukumar\TIU Materials\Machine Learning\Odd Semester 2024\Lab Experiments\Assignment-9\IRISFlower_Data.csv")
print(df.head())
print(df.describe())

#%% Encode the class labels
#LabelEncoder encodes the labels to 0, 1, 2
#fit_transform() returns the array of labels and those values are put into the data frame column

le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
print(df.head())

#%% Normalize the data

#StandardScaler standardize features by removing the mean and scaling to unit variance.
#The standard score of a sample x is calculated as z = (x - u) / s
#Here it transforms the first 4 columns
sc = StandardScaler()
df.iloc[:,[0, 1, 2, 3]] = sc.fit_transform(df.iloc[:,[0, 1, 2, 3]])
print(df)

#%% Create the training and testing datasets

#Create X as the data and Y as the label
X = df.drop('Class', axis=1)
Y = df['Class']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#%% Defibe your own function to compute accuracy

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#%% Fit the model, on training data and evaluate its performance on test data

classifier = MLPClassifier(hidden_layer_sizes=(10),
                           max_iter=3000,
                           activation = 'relu',
                           solver='sgd',
                           learning_rate = 'constant',
                           verbose = True,
                           learning_rate_init = 0.001,
                           random_state=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, Y_test)
print("The Confusion Matrix\n-----------------------")
print(cm)
print("Accuracy of MLPClassifier : ", accuracy(cm))

#Print the connection weights

print(classifier.coefs_)