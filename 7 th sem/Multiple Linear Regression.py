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
