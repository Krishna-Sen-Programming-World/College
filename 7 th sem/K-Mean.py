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
