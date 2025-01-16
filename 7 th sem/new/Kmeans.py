import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#%% Prepare the data set with 500 2 dimensional data points distributed in 4 classes, and plot it

N = 50
DIM = 2
C = 3

X,y = make_blobs(n_samples = N, n_features = DIM, centers = C, random_state = 10)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()

#%% Assume k=4, initialize 4 random cluster centroids and plot them along with the data points

K = 3

x_min = min(X[:,0])
print(x_min)
x_max = max(X[:,0])
print(x_max)
x_range = x_max - x_min
print(x_min, x_max, x_range)

y_min = min(X[:,1])
print(y_min)
y_max = max(X[:,1])
print(y_max)
y_range = y_max - y_min
print(y_min, y_max, y_range)

centers = np.zeros((K, 2))

for i in range (K):
    x_val = x_min + (i+1)*x_range/(K+1)
    y_val = y_min + (i+1)*y_range/(K+1)
    centers[i][0] = x_val
    centers[i][1] = y_val

plt.scatter(centers[:,0], centers[:,1])
plt.scatter(X[:,0],X[:,1])
plt.show()
print(centers)

#%% Iteratively Assign the data points to one of the four clusters based on their distances from the centroids

shifted_count = N
cluster = np.zeros(N)
for i in range(N):
    cluster[i] = 0

iteration_count = 0
    
while (shifted_count > 0):
    
    shifted_count = 0
    iteration_count += 1
    
    for i in range (N):
        min_distance = float('inf')
        for j in range (K):
            distance = (X[i][0] - centers[j][0])**2 + (X[i][1] - centers[j][1])**2
            if(distance < min_distance):
                min_distance = distance
                best_cluster = j
        if(cluster[i] != best_cluster):
            shifted_count += 1
            cluster[i] = best_cluster
            
#    print(cluster)
    print("Shift Count:" + str(shifted_count))
    print("Iteration Count:" + str(iteration_count))
    
#Now shift the centers according to the distribution of data points

    for i in range (K):
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        for j in range (N):
            data_x = X[j][0]
            data_y = X[j][1]
            if(cluster[j] == i):
                if(data_x <= x_min):
                    x_min = data_x
                if(data_x >= x_max):
                    x_max = data_x
                if(data_y <= y_min):
                    y_min = data_y
                if(data_y >= y_max):
                    y_max = data_y
                        
        centers[i][0] = (x_max + x_min)/2
        centers[i][1] = (y_max + y_min)/2
        
    print(centers)
    plt.scatter(centers[:,0], centers[:,1])
    plt.scatter(X[:,0],X[:,1])
    plt.show()



print("DONE")