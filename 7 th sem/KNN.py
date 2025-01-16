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
