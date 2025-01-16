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
