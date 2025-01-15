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



