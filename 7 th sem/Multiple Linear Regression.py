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
