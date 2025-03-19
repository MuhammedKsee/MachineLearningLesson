from sklearn.neural_network import MLPClassifier
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

mlp = MLPClassifier(hidden_layer_sizes = (2,),activation = "relu",max_iter=10000)

mlp.fit(x,y)

y_pred = mlp.predict(x)

accuracy = np.mean(y_pred == y)
print(accuracy)

