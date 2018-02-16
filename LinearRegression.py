# Linear Regression from scratch

import numpy as np
import pandas as pd

np.random.seed(0)

features = 10
trainingSize = 10 ** 2

randData = np.random.rand(trainingSize, features + 1)
colNames = [f'f{i}' for i in range(1, features + 1)]
colNames.append('labels')

df = pd.DataFrame(randData, columns=colNames)

data = df.iloc[:, :-1]
dummy_feature = pd.Series(np.ones(trainingSize), name='f0')

X = pd.concat([dummy_feature, data], axis=1)
print(X.tail())
y = df['labels']
thetas = np.random.rand(features + 1)

y_hats = lambda X, thetas: np.matmul(X, thetas)
print(y_hats(X, thetas).shape)
