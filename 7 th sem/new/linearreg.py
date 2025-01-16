from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LinearRegression

#%%

#read the data into a dataframe and concert it into a numpy array
df = pd.read_csv(r"D:\Sukumar\TIU Materials\Machine Learning\Sign_Distance_Dataset\sign_distance_dataset.csv")
dst_data = df.to_numpy()

#print the data to confirm the read
print(np.shape(dst_data))
x = dst_data[:,0]
y = dst_data[:,1]
print("age:" + str(x))
print("distance:" + str(y))

#%%

#calculate the required parameters sum(xy), sum(x^2), sum(x) and sum(y)
N = len(dst_data[:,0])
print("data count: " + str(N))
xy = x * y
x2 = x * x

xy_sum = sum(xy)
x2_sum = sum(x2)
x_sum = sum(x)
y_sum = sum(y)

print("xy_sum, x2_sum, x_sum, y_sum = " + str(xy_sum) + ", " + str(x2_sum) + ", " + str(x_sum) + ", " + str(y_sum))

#%%

#calculate m and c of the regression line
M = (N*xy_sum - x_sum*y_sum)/(N*x2_sum - x_sum*x_sum)
print("M: " + str(M))

C = (y_sum - M * x_sum)/N
print("C: " + str(C))

#plot the line and the points
Y = M*x + C

plt.plot(x, Y)
plt.scatter(x, y)
plt.show()

#%%

#Find out the value of y for an unknown value of x
x_unk = 50
y_unk = M*x_unk + C
print("Predicted y value for x=50: " + str(y_unk))

x_unk = 90
y_unk = M*x_unk + C
print("Predicted y value for x=90: " + str(y_unk))

#Now we will use python libraries to do the regression analysis on the same data set
model = LinearRegression().fit(x.reshape(-1,1), y)
intercept = model.intercept_
slope = model.coef_[0]
print("intercept and slope: " + str(intercept) + ", " + str(slope))

#Get the R^2 value
R_squared_value = model.score(x.reshape(-1,1), y)
print("R^2 value: " + str(R_squared_value))

#Predict the y value for an unknown x value
x_unknown = np.array([50])
y_pred = model.predict(x_unknown.reshape(-1,1))
print("y_pred = " + str(y_pred))
