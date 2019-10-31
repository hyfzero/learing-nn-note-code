import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing data
print('Importing data...')
data = pd.read_csv('ex1data1.txt', sep=',', names=['population', 'profit'])
X = data['population'].values
y = data['profit'].values

plt.title("linear regression demo")
plt.xlabel("years")
plt.ylabel("hosing price")
plt.plot(X, y, "ob")
plt.show()
print(X.shape)
print(y.shape)