## Code 1.1 Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('I:/data.csv')

# Drop the first row if necessary (based on your requirement)
df = df.drop(index=0)

# Check the first few rows of the dataset
print(df.head())

# Select features and target variable
x = df.iloc[:, :-1].values  # All rows, all columns except the last one (YearsExperience)
y = df.iloc[:, -1].values   # Last column as the target (Salary)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Create and train the linear regression model
regg = LinearRegression()
regg.fit(x_train, y_train)

# Make predictions
Y_pred = regg.predict(x_test)

# Print the test set and corresponding predicted values
print("Test Data (x_test)  Predicted Values (Y_pred):")
print(np.column_stack((x_test, Y_pred)))

# Calculate R-squared score
res = r2_score(y_test, Y_pred)
print(f"R-squared score: {res * 100:.2f}%")

# Plot the results
plt.scatter(x_train, y_train, color='red', label='Training Data')  # Training data points
plt.scatter(x_test, y_test, color='blue', label='Testing Data')    # Testing data points
plt.plot(x_test, Y_pred, color='green', label='Regression Line')   # Regression line
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression: Salary vs Years of Experience')
plt.legend()
plt.show()



'''
YearsExperience,Salary
1.1,39343.00
1.3,46205.00
1.5,37731.00
2.0,43525.00
2.2,39891.00
2.9,56642.00
'''
# Code 1.2 Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt

def estimate_coeff(p, q):
    # Estimate the total number of points or observations
    n1 = np.size(p)
    
    # Calculate the mean of the p and q vectors
    m_p = np.mean(p)
    m_q = np.mean(q)
    
    # Calculate the cross deviation and deviation about p
    SS_pq = np.sum(q * p) - n1 * m_q * m_p
    SS_pp = np.sum(p * p) - n1 * m_p * m_p
    
    # Calculate the regression coefficients
    b_1 = SS_pq / SS_pp
    b_0 = m_q - b_1 * m_p
    
    print(f"The intercept c is {b_0}, the slope m is {b_1}")
    
    # Calculate the value for the prediction when x = 9
    ans = (b_1 * 9) + b_0
    print(f"For x = 9, the predicted y value is {ans}")
    
    return (b_0, b_1)

def plot_regression_line(p, q, b):
    # Plot the actual points or observations as a scatter plot
    plt.scatter(p, q, color="m", label="Observed Points", marker="o", s=30)
    
    # Calculate the predicted response vector using the regression equation
    q_pred = b[0] + b[1] * p
    
    # Plot the predicted points
    plt.scatter(p, q_pred, color="b", label="Predicted Points", marker="x", s=40)
    
    # Plot the regression line
    p_line = np.linspace(np.min(p), np.max(p), 100)  # Create more points for a smooth line
    q_line = b[0] + b[1] * p_line  # Predicted y-values for these points
    
    plt.plot(p_line, q_line, color="g", label="Regression Line")
    
    # Adding labels and title
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('Linear Regression')
    plt.legend()
    
    # Show the plot
    plt.show()

def main():
    # Input the observation points or data
    p = np.array([0,1,2,3,3,5,5,5,6,7,7,10])
    q = np.array([96,85,82,74,95,68,76,84,58,65,75,50])
    
    # Estimate the coefficients
    b = estimate_coeff(p, q)
    print("Estimated coefficients are:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))
    
    # Plot the regression line
    plot_regression_line(p, q, b)

if __name__ == "__main__":
    main()







## Code 2 Multiple Linear Regression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score

df=pd.read_csv('I:\data.csv')
print(df.head())
sns.scatterplot(data=df, x='R&D Spend', y='Profit', hue='State', size='Marketing Spend', alpha=0.7)
plt.show()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(x.shape)
print(y.shape)
states = pd.get_dummies(x['State'], drop_first = True)
x = x.drop('State',axis=1)
x = pd.concat([x,states],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
regg = LinearRegression()
regg.fit(x_train, y_train)
Y_pred=regg.predict(x_test)
print(x_test,Y_pred)
res=r2_score(Y_pred,y_test)
print('Model Accuracy :',res*100,'%')


# Partial Dependence Plot for R&D Spend
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(regg, x_train, features=['R&D Spend'], ax=ax)
plt.title('Multiple Regression Line')
plt.show()


# Example prediction code
new_data = np.array([[12000, 95000,50000,0, 1]])
predicted_profit = regg.predict(new_data)
print("Predicted Profit:", predicted_profit[0])
'''
R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.17,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
114523.61,122616.84,261776.23,New York,129917.04
'''


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

##Code 5 Decision Tree
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


## Code 6 KNN
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt 

df=pd.read_csv('I:\data.csv')
print(df.head())
df.info()
print(df.describe())
y = df['Last'] 
X = df.drop('Last', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
'''
knn = KNeighborsClassifier(n_neighbors = 6) 
knn.fit(X_train, y_train) 
knn_score = knn.score(X_test, y_test) 
print(knn_score)
'''
best_k = 0
best_score = 0
s = len(X_train) - 1
s=min(s, len(X_train))
for k in range(2, s):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    current_score = knn.score(X_test, y_test)
    
    if current_score > best_score:
        best_score = current_score
        best_k = k
print(f"For {best_k} neighbour, acuracy is {best_score*100:.2f}%")

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Length', y='Weight', hue='Last', alpha=0.7)
plt.show()
#or 
import plotly.express as px
fig=px.scatter_3d(df, x='Length', y='Weight', z='Cost', color ='Last')
fig.show()

'''
Length,Weight,Cost,Category,Last
10,15,45,1,High
11,6,37,2,Medium
12,14,48,3,High
7,9,33,4,Low
'''

##Code 7 K-mean
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

old_df=pd.read_csv('I:\data.csv')
print(old_df.head())
old_df.info()
print(old_df.describe())
df=old_df[['ApplicantIncome', 'LoanAmount']]
print(df.head())
df.info()

plt.scatter(df['ApplicantIncome'],df['LoanAmount'])
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.show()

sse = [] #SUM OF SQUARED ERROR
for k in range(1,20):
    km = KMeans(n_clusters=k)
    km.fit_predict(df)
    sse.append(km.inertia_)
print(sse)
plt.plot(range(1,20), sse)
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.show()

X = df.iloc[:,:].values
km = KMeans(n_clusters = 10)
y_means=km.fit_predict(X)

colors = ['red', 'blue', 'green', 'yellow', 'black', 
    'magenta', 'purple', 'grey', 'pink', 'brown']

# Plotting Function
for i in range(len(y_means)):
    plt.scatter(
        X[y_means == i, 0],   # X-coordinate
        X[y_means == i, 1],   # Y-coordinate
        color=colors[i % len(colors)]  # Cycle through colors
    )

# Plot Cluster Centers
plt.scatter(
    km.cluster_centers_[:, 0], 
    km.cluster_centers_[:, 1], 
    color='red', 
    marker='x', 
    s=200
)

plt.title('K-Means Clustering')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.show()
'''
Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
LP001003,Male,Yes,1,Graduate,No,4583,1508.0,128.0,360.0,1.0,Rural,N
LP001005,Male,Yes,0,Graduate,Yes,3000,0.0,66.0,360.0,1.0,Urban,Y
LP001006,Male,Yes,0,Not Graduate,No,2583,2358.0,120.0,360.0,1.0,Urban,Y
LP001008,Male,No,0,Graduate,No,6000,0.0,141.0,360.0,1.0,Urban,Y
LP001013,Male,Yes,0,Not Graduate,No,2333,1516.0,95.0,360.0,1.0,Urban,Y
LP001024,Male,Yes,2,Graduate,No,3200,700.0,70.0,360.0,1.0,Urban,Y
LP001027,Male,Yes,2,Graduate,,2500,1840.0,109.0,360.0,1.0,Urban,Y
LP001029,Male,No,0,Graduate,No,1853,2840.0,114.0,360.0,1.0,Rural,N
LP001030,Male,Yes,2,Graduate,No,1299,1086.0,17.0,120.0,1.0,Urban,Y
LP001032,Male,No,0,Graduate,No,4950,0.0,125.0,360.0,1.0,Urban,Y
LP001034,Male,No,1,Not Graduate,No,3596,0.0,100.0,240.0,,Urban,Y
LP001036,Female,No,0,Graduate,No,3510,0.0,76.0,360.0,0.0,Urban,N
LP001038,Male,Yes,0,Not Graduate,No,4887,0.0,133.0,360.0,1.0,Rural,N
LP001041,Male,Yes,0,Graduate,,2600,3500.0,115.0,,1.0,Urban,Y
LP001043,Male,Yes,0,Not Graduate,No,7660,0.0,104.0,360.0,0.0,Urban,N
LP001047,Male,Yes,0,Not Graduate,No,2600,1911.0,116.0,360.0,0.0,Semiurban,N
LP001050,Male,Yes,2,Not Graduate,No,3365,1917.0,112.0,360.0,0.0,Rural,N
LP001068,Male,Yes,0,Graduate,No,2799,2253.0,122.0,360.0,1.0,Semiurban,Y
LP001073,Male,Yes,2,Not Graduate,No,4226,1040.0,110.0,360.0,1.0,Urban,Y
LP001086,Male,No,0,Not Graduate,No,1442,0.0,35.0,360.0,1.0,Urban,N
LP001087,Female,No,2,Graduate,,3750,2083.0,120.0,360.0,1.0,Semiurban,Y
LP001095,Male,No,0,Graduate,No,3167,0.0,74.0,360.0,1.0,Urban,N
LP001097,Male,No,1,Graduate,Yes,4692,0.0,106.0,360.0,1.0,Rural,N
'''


###problem 2  (K-Mean Clusstering)
'''
Write a python program to implement the K-Means clustering algorithm, as per the following details
a. Use the 'make_blobs' function to create a 2-dimensional data set containing 1000 data points,
b. Number of clusters is the number of zeros in your ID
c. Display the dataset using a scatter plot [output-1]-
d. Initialize the centroids as the equal partitioning points of the range along each dimension le, if the minimum and maximum values along a dimension are -1 and +15 then the centroids will be placed at 3, 7 and 11.
e. Plot the initial position of the centroids [output-2]
f. Keep on iterating until not a single data point is getting shifted from one cluster to other.
g. Plot the initial position of the centroids [output-3]
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Step 1: Create a 2D dataset with 1000 data points
n_samples = 1000
X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)

# Step 2: Number of clusters (replace 'your_id' with your actual ID)
# For example, if your ID has two zeros, set n_clusters = 2
n_clusters = 2  # Change this according to the number of zeros in your ID

# Step 3: Display the dataset using a scatter plot
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title('Generated Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 4: Initialize centroids
min_val = X.min(axis=0)
max_val = X.max(axis=0)
centroids = np.linspace(min_val, max_val, n_clusters)

# Step 5: Plot the initial position of the centroids
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=200)
plt.title('Initial Centroid Positions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 6: K-Means Algorithm
def k_means(X, centroids):
    while True:
        # Assign clusters based on closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Calculate new centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return labels, centroids

# Run K-Means
final_labels, final_centroids = k_means(X, centroids)

# Step 7: Plot final clusters and centroids
colors = ['red', 'blue', 'green', 'yellow', 'black', 
          'magenta', 'purple', 'grey', 'pink', 'brown']

for i in range(n_clusters):
    plt.scatter(X[final_labels == i, 0], X[final_labels == i, 1], color=colors[i % len(colors)], s=30)

# Plot final centroids
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='black', marker='x', s=200)
plt.title('Final Clusters and Centroid Positions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



###(Local minima) Gradient Descent.py
'''
Write a python program to find a local minima by using gradient descent technique, as per the following
details: a. The function to use is y = x^2 - 4x + C, where C is the sum of the digits of your ID, e.g., if your ID is
211001001176, then the value of C = 20. 
b. Plot the graph of the above function. For plotting, the initial data points can be taken as x = [-4,-3,- 2, -1, 0, 1, 2, 3, 4, 5, 6, 7] [output-1]
c. Plot the results of the following runs:
i. start_point(S) = 10, lambda(L) = 0.01, no. of iterations(N) = 500 [output-2] ii. start_point(S) = 10, lambda(L) = 0.1, no. of iterations(N) = 500 [output-3
'''
Solution:

import numpy as np
import matplotlib.pyplot as plt

# User ID (replace with your actual ID)
user_id = "211001001176"  # Example ID
C = sum(map(int, user_id))  # Sum of digits of ID

# Define the function y = x^2 - 4x + C and its derivative
def func(x):
    return x**2 - 4*x + C

def derivative(x):
    return 2*x - 4

# Step b: Plot the function
x_values = np.linspace(-4, 7, 100)
y_values = func(x_values)

plt.plot(x_values, y_values, label="y = x^2 - 4x + C")
plt.scatter(np.arange(-4, 8), func(np.arange(-4, 8)), color="red", label="Initial Points")
plt.title("Function Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()  # Output-1

# Gradient Descent Implementation
def gradient_descent(start_point, learning_rate, iterations):
    x = start_point
    path = [x]
    for _ in range(iterations):
        grad = derivative(x)
        x -= learning_rate * grad
        path.append(x)
    return x, path

# Parameters for runs
runs = [
    {"S": 10, "L": 0.01, "N": 500, "label": "Run 1: L=0.01, N=500"},
    {"S": 10, "L": 0.1, "N": 500, "label": "Run 2: L=0.1, N=500"}
]

# Perform Gradient Descent for each run
for i, run in enumerate(runs, start=2):
    final_x, path = gradient_descent(run["S"], run["L"], run["N"])
    path_y = func(np.array(path))
    
    # Plot the results
    plt.plot(x_values, y_values, label="y = x^2 - 4x + C")
    plt.scatter(path, path_y, color="orange", s=10, label="Gradient Descent Path")
    plt.scatter(final_x, func(final_x), color="red", label="Local Minima")
    plt.title(f"Gradient Descent Results ({run['label']})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()  # Output-2, Output-3 (as per runs)


### Linear Regression(0)
'''
bda(L) 0

a. Generate 10 random values between 1 to 50 as x1, x2,..., x10

b. Generate 10 random values between 100 to 200 as y1, y2, ..., y10

c. Plot the points (x1, y1), (x2, y2), ..., (x10, y10) [output-1]

d. Fit a regression line using the library of scikit_learn and plot the line along with the points [output-2]

e. Display the y value of a data point whose x value is 60 [output-3]
'''
Solution:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# a. Generate 10 random values between 1 to 50 as x1, x2, ..., x10
x = np.random.randint(1, 51, 10)

# b. Generate 10 random values between 100 to 200 as y1, y2, ..., y10
y = np.random.randint(100, 201, 10)

# c. Plot the points (x1, y1), (x2, y2), ..., (x10, y10)
plt.scatter(x, y, color='blue', label='Data points')
plt.title('Scatter Plot of Data Points')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()

# d. Fit a regression line using scikit-learn and plot the line along with the points
x_reshaped = x.reshape(-1, 1)  # Reshape x to be a 2D array for the model
model = LinearRegression()
model.fit(x_reshaped, y)

# Predict y values using the model
y_pred = model.predict(x_reshaped)

# Plotting the points and the regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Regression Line and Data Points')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()

# e. Display the y value for a data point with x = 60
x_new = np.array([[60]])  # New data point for which we want to predict y
y_new = model.predict(x_new)

# Output the predicted y value
print(f"The predicted y value for x = 60 is: {y_new[0]}")



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


### SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# Step 1: Create a 2D dataset with 1000 data points using make_blobs
# Number of clusters is determined by the number of zeros in your ID. For example, if your ID has two zeros:
n_clusters = 5  # Change this according to the number of zeros in your ID
X, y = make_blobs(n_samples=1000, centers=n_clusters, n_features=2, random_state=42)

# Step 2: Display the dataset using a scatter plot [output-1]
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='viridis')
plt.title('Generated Dataset using make_blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Train the SVM model with a linear kernel
clf_linear = SVC(kernel='linear')
clf_linear.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred_linear = clf_linear.predict(X_test)
accuracy_linear = accuracy_score(y_pred_linear, y_test)
print(f"Linear Kernel Accuracy: {accuracy_linear:.2f}")

# Step 6: Visualization of results with Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', alpha=0.7)
plt.title('Weight vs Length1 by Class')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Cluster Label')
plt.show()


### KNN
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

# Step 1: Generate synthetic dataset
X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42)

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=['Length', 'Weight'])
df['Last'] = y  # Assign cluster labels as the target variable

# Display the first few rows of the DataFrame
print(df.head())
df.info()
print(df.describe())

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Length', 'Weight']], df['Last'], test_size=0.2, random_state=3)

# Step 3: Implement KNN and find the best k
best_k = 0
best_score = 0
s = len(X_train) - 1
s = min(s, len(X_train))

for k in range(2, s):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    current_score = knn.score(X_test, y_test)
    
    if current_score > best_score:
        best_score = current_score
        best_k = k

print(f"For {best_k} neighbors, accuracy is {best_score * 100:.2f}%")

# Step 4: Visualize the data using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Length', y='Weight', hue='Last', alpha=0.7)
plt.title('Seaborn Scatter Plot of Generated Data')
plt.show()

# Step 5: Visualize the data using Plotly (3D scatter plot)
fig = px.scatter_3d(df, x='Length', y='Weight', z='Last', color='Last')
fig.update_layout(title='Plotly 3D Scatter Plot of Generated Data')
fig.show()

### Decissio Tree
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


