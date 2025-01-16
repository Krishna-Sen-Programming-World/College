from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

#df = pd.read_csv(r"D:\Sukumar\TIU Materials\Machine Learning\IrisFlower_Dataset\IRISFlower_Data.csv")
#Prepare the Dataframe
#Load the data
iris= load_iris()

#Convert it into a dataframe
df = pd.DataFrame(
    iris.data, 
    columns=iris.feature_names
    )

#Add the class column
df['class'] = iris.target

# Map targets to target names
target_names = {
    0:'setosa',
    1:'versicolor', 
    2:'virginica'
}

#Convert the 'target' column from 0, 1, 2 to actual class labels
df['class'] = df['class'].map(target_names)

DIM = 4

#Display the first few rows
print(df.head())
#%%

#function for computing the distance between two data points
def norm_2(data_1, data_2) :
    norm = 0
    for i in range (4) :
        norm = norm + (data_1[i] - data_2[i])**2
    return norm

#%%

total_correct_count = 0
total_false_count = 0
for iteration_count in range(5) :
    #Split training and testing data sets
    df_trng = df.sample(frac = 0.8)
    df_tst = df.drop(df_trng.index)

    #Compute the NN performance of the training set
    correct_count = 0
    false_count = 0
    for i in range(len(df_tst)) :
        tst_data = df_tst.iloc[i]
        least_distance = 9999
        nearest_data = df_trng.iloc[0]
        for j in range(len(df_trng)) :
            trng_data = df_trng.iloc[j]
            distance = norm_2(tst_data, trng_data)
            if(distance < least_distance) :
                least_distance = distance
                nearest_data = trng_data
        
        computed_lbl = nearest_data[DIM]
        actual_lbl = tst_data[DIM]
        if(computed_lbl == actual_lbl) :
            correct_count = correct_count + 1
        else :
            false_count = false_count + 1

    print("Correct Count: " + str(correct_count))
    print("False Count: " + str(false_count))
    total_correct_count = total_correct_count + correct_count
    total_false_count = total_false_count + false_count
    
avg_accuracy = total_correct_count / (total_correct_count + total_false_count)
print("Average Accuracy: " + str(avg_accuracy))
print("***********************************************")

#%%

X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

score_sheet = []
index = []
for i in range (10) :

    #Fit the data into the model
    knn = KNeighborsClassifier(n_neighbors=i+1)
    knn.fit(X_train, y_train)

    #Do the prediction and get the score
    score = knn.score(X_test, y_test)
    print("Score:" + str(score))
    index.append(i+1)
    score_sheet.append(score)

plt.plot(index, score_sheet)
plt.show()