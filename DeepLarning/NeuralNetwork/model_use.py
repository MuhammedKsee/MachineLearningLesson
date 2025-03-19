from simpleNeuralNetwork import MultiLayerPropagations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

my_mlp = MultiLayerPropagations(2, 4, 1)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

my_mlp.fit(x,y)

predict = my_mlp.predict(x)

print(predict)

accuracy = (predict == y).mean()

print(accuracy)

plt.figure()
plt.imshow(predict.reshape())