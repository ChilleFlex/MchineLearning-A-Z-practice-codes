import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the datasets from a file
dataset = pd.read_csv('Position_Salaries(for_regression_practices).csv')       

#making of x and y 
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values
y = y.reshape(len(y),1)



#spliting into training set and test set
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#fitting SVR model to the dataset
#create regression model here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

#predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))


#visualising regression results
plt.scatter(x, y, color = 'blue')
plt.plot(x, regressor.predict(x), color = 'red')
plt.title('truth or bluff(SVR model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

"""
#visualising regression results for higher resolution
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = 'black')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('truth or bluff(regression model)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
"""



