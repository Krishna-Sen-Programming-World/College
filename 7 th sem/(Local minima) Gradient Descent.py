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