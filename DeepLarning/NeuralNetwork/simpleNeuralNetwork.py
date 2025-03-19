import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivate(z):
    return sigmoid(z)*(1-sigmoid(z))

class MultiLayerPropagations:
    def __init__(self,input_size,hidden_size,output_size,learning_rate = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weight1 = np.random.randn(self.input_size,self.hidden_size)
        self.weight2 = np.random.randn(self.hidden_size,self.output_size)

        self.bias1 = np.zeros((1,self.hidden_size))
        self.bias2 = np.zeros((1,self.output_size))

    def fit(self,x,y,epochs=1000):
        for i in range(epochs):
            # fit forward #
            layer1 = x.dot(self.weight1) + self.bias1
            activation = sigmoid(layer1)

            layer2 = activation.dot(self.weight2) + self.bias2
            output = sigmoid(layer2)
            error = output - y

            # back propagations # 
            d_weight2 = activation.T.dot(error * sigmoid_derivate(layer2))
            d_bias2 =   np.sum(error*sigmoid_derivate(layer2),axis=0,keepdims=True)

            error_hidden = error.dot(self.weight2.T) * sigmoid_derivate(layer1)

            d_weight1 = x.T.dot(error_hidden)
            d_bias1 = np.sum(error_hidden,axis=0,keepdims=True)

            # update weights and biases #
            self.weight1 -= self.learning_rate * d_weight1
            self.bias1 -= self.learning_rate * d_bias1
            self.weight2 -= self.learning_rate * d_weight2
            self.bias2 -= self.learning_rate * d_bias2

            Loss = np.mean(error **2) 

            if i %100 == 0:
                print("Loss :",Loss)


    def predict(self,x):
        layer1 = x.dot(self.weight1) + self.bias1
        activation = sigmoid(layer1)

        layer2 = activation.dot(self.weight2) + self.bias2
        output = sigmoid(layer2)
        
        return (output>=0.5).astype(int)



        

 