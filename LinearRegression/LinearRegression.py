# Performs Linear Regression (from scratch) using randomized data
# Optimizes weights by using Gradient Descent Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

features = 3
trainingSize = 10 ** 1
trainingSteps = 10 ** 3
learningRate = 10 ** -2

randData = np.random.rand(trainingSize, features + 1)
colNames = [f'f{i}' for i in range(1, features + 1)]
colNames.append('labels')

dummy_column = pd.Series(np.ones(trainingSize), name='f0')
df = pd.DataFrame(randData, columns=colNames)

X = pd.concat([dummy_column, df.drop(columns='labels')], axis=1)
y = df['labels']
thetas = np.random.rand(features + 1)

cost = lambda thetas: 0.5 * np.mean((np.matmul(X, thetas) - y) ** 2)
dJdtheta = lambda thetas, k: np.mean((np.matmul(X, thetas) - y) * X.iloc[:, k])
gradient = lambda thetas: np.array([dJdtheta(thetas, k) for k in range(X.shape[1])])

# J(theta) before gradient descent
print(cost(thetas))

# Perform gradient descent
errors = np.zeros(trainingSteps)
for step in range(trainingSteps):
    thetas -= learningRate * gradient(thetas)
    errors[step] = cost(thetas)

# J(theta) after gradient descent
print(cost(thetas))

# Plots Cost function as gradient descent runs
plt.plot(errors)
plt.xlabel('Training Steps')
plt.ylabel('Cost Function')
plt.show()
