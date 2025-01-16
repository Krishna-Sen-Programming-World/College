from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

#We are trying to apply gradient descent to minimize the function y = x^2 - 3*x + 20
#initialize the data
data = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
x = np.array(data)
y = np.array(x**2 - 3*x + 20)


#plot the points and the graphs
Y = x**2 - 3*x + 20
plt.plot(x, Y)
plt.scatter(x, y)
plt.show()

#We know that the gradient will be 2x - 3
#So, we will start from a random guess and see how does it converge
#We first start from 10, and our learning rate is 0.01
#We run for 500 iterations
last_guess = 10
last_result = last_guess**2 - 3*last_guess + 20
print("-" + "\tGuess: " + str(last_guess) + "\tResult: " + str(last_result))

guesses = []
guesses.append(last_guess)
results = []
results.append(last_result)
learning_rate = 0.01
for i in range (500) :
    correction = learning_rate*(2*last_guess - 3)
    new_guess = last_guess - correction
    guesses.append(new_guess)
    new_result = new_guess**2 - 3*new_guess + 20
    results.append(new_result)
    print(str(i) + "\tGuess: " + str(new_guess) + "\tResult: " + str(new_result))
    last_guess = new_guess

plt.scatter(guesses, results)
plt.title("lambda = 0.01")
plt.show()

#Now let us see the impact of the learning rate
#First we will try with lambda = 0.1 and run for 100 iterations

last_guess = 10
last_result = last_guess**2 - 3*last_guess + 20
print("-" + "\tGuess: " + str(last_guess) + "\tResult: " + str(last_result))

guesses = []
guesses.append(last_guess)
results = []
results.append(last_result)
learning_rate = 0.1
for i in range (100) :
    correction = learning_rate*(2*last_guess - 3)
    new_guess = last_guess - correction
    guesses.append(new_guess)
    new_result = new_guess**2 - 3*new_guess + 20
    results.append(new_result)
    print(str(i) + "\tGuess: " + str(new_guess) + "\tResult: " + str(new_result))
    last_guess = new_guess

plt.scatter(guesses, results)
plt.title("lambda = 0.1")
plt.show()

#%%Now we will try with lambda = 1.0 and run for 100 iterations

last_guess = 10
last_result = last_guess**2 - 3*last_guess + 20
print("-" + "\tGuess: " + str(last_guess) + "\tResult: " + str(last_result))

guesses = []
guesses.append(last_guess)
results = []
results.append(last_result)
learning_rate = 1
for i in range (100) :
    correction = learning_rate*(2*last_guess - 3)
    new_guess = last_guess - correction
    guesses.append(new_guess)
    new_result = new_guess**2 - 3*new_guess + 20
    results.append(new_result)
    print(str(i) + "\tGuess: " + str(new_guess) + "\tResult: " + str(new_result))
    last_guess = new_guess

plt.scatter(guesses, results)
plt.title("lambda = 1.0")
plt.show()

# Now we will try with lambda = 0.5 and run for 100 iterations

last_guess = 10
last_result = last_guess**2 - 3*last_guess + 20
print("-" + "\tGuess: " + str(last_guess) + "\tResult: " + str(last_result))

guesses = []
guesses.append(last_guess)
results = []
results.append(last_result)
learning_rate = 0.5
for i in range (100) :
    correction = learning_rate*(2*last_guess - 3)
    new_guess = last_guess - correction
    guesses.append(new_guess)
    new_result = new_guess**2 - 3*new_guess + 20
    results.append(new_result)
    print(str(i) + "\tGuess: " + str(new_guess) + "\tResult: " + str(new_result))
    last_guess = new_guess

plt.scatter(guesses, results)
plt.title("lambda = 0.5")
plt.show()

# Now we will try with lambda = 0.75 and run for 100 iterations

last_guess = 10
last_result = last_guess**2 - 3*last_guess + 20
print("-" + "\tGuess: " + str(last_guess) + "\tResult: " + str(last_result))

guesses = []
guesses.append(last_guess)
results = []
results.append(last_result)
learning_rate = 0.75
for i in range (100) :
    correction = learning_rate*(2*last_guess - 3)
    new_guess = last_guess - correction
    guesses.append(new_guess)
    new_result = new_guess*new_guess - 3*new_guess + 20
    results.append(new_result)
    print(str(i) + "\tGuess: " + str(new_guess) + "\tResult: " + str(new_result))
    last_guess = new_guess

plt.scatter(guesses, results)
plt.title("lambda = 0.75")
plt.show()