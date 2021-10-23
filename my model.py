import numpy as np
from optimizers import *
from Dataset import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class Model_Classification:
    def __init__(self, lr, epoches, optimizer, batch_size=32, activation='tanh', hidden_size=[100, 1]):
        self.lr = lr
        self.batch_size = batch_size
        self.epoches = epoches
        self.optimizer = optimizer
        self.dataset = Dataset(32)
        self.activation = activation
        self.hidden_size=hidden_size
        self.w, self.bias = self.initilization_weights(self.dataset.X_train.shape[1], hidden_size)
        self.L = len(hidden_size)

    def initilization_weights(input_size,layer_size):
        weights = [0]*len(layer_size)
        bias = [0]*len(layer_size)
        np.random.seed(0) 
        weights[0] = np.random.randn(layer_size[0], input_size) * np.sqrt(2 / input_size)
        bias[0] = np.zeros((layer_size[0], 1))
        for l in range(1,len(layer_size)):
            weights[l] = np.random.randn(layer_size[l],layer_size[l-1]) * np.sqrt(2/layer_size[l])
            bias[l] = np.zeros((layer_size[l],1))
        return weights, bias

    def predict(self, X):
        A = X
        for l in range(self.L-1):
            w = self.w[l]
            b = self.bias[l]
            A = self.forward_activation(A, w, b, 'relu')
        w = self.w[self.L-1]
        b = self.bias[self.L-1]
        A = self.forward_activation(A, w, b, 'sigmoid')
        Y_predict = (A > 0.5)
        return Y_predict
    
    def model_forward(self, X, keep_prob=1):
        cache_D = []
        cache_A = []
        A = X
        L=self.hidden_size

        for l in range(L-1):
            w = self.w[l]
            b = self.bias[l]
            A = self.forward_activation(A, w, b, 'relu')
            if l%2 == 0:
                cache_D[l] = np.random.randn(A.shape[0],A.shape[1]) < keep_prob
                A = A * cache_D[l] / keep_prob
            cache_A[l] = A
        w = self.w['W' + str(L-1)]
        b = self.b['b' + str(L-1)]
        A = self.forward_activation(A, w, b, 'sigmoid')
        cache_A[l-1] = A
        return cache_D, cache_A, A

    def forward_activation(self, X, activation):
        z = np.dot(X,self.w.T) + self.bias.T
        if activation == 'relu':
            A = np.maximum(0, z)
        elif activation == 'sigmoid':
            A = 1/(1+np.exp(-z))
        else:
            A = np.tanh(z)
        return A

    def predict(self, X, w):
        return X @ self.w

    def backward(self,X, Y, cache_D, cache_A,keep_prob=1):
        grad =[]
        m = Y.shape[0]

        cache_A['pred'] = X
        grad['dz' + str(self.L-1)] = cache_A[self.L-1] - Y
        cache_D['pred'] = 0
        for l in reversed(range(self.L)):
            grad['dW' + str(l)] = (1 / m) * np.dot(grad['dz' + str(l)].T, cache_A[l-1])
            grad['db' + str(l)] = 1 / m * np.sum(grad['dz' + str(l)].T, axis=1, keepdims=True)
            if l%2 != 0:
                grad['dz' + str(l-1)] = ((np.dot(grad['dz' + str(l)], self.w[l]) * cache_D[l-1] / keep_prob) *
                                    cache_A[l-1] > 0)
            else:
                grad['dz' + str(l - 1)] = (np.dot(grad['dz' + str(l)], self.w[l]) *
                                        cache_A[l - 1] > 0)
        return grad
    def cost_f(Y_pred, Y):
        m = Y.shape[0]
        cost = -1/m * np.sum(Y * np.log(Y_pred ) + (1-Y) * np.log(1-Y_pred))
        return cost
    def mean_squared_error(y_pred, y):
            return sum((y_pred-y)**2)/y.shape
    def train(self):
        opt = None
        losses = []
        if self.optimizer == 'momentum':
            opt = Momentum(self.w,self.lr)

        elif self.optimizer == 'rmsprop':
            opt = RMS_Prop(self.w,self.lr)

        elif self.optimizer == 'adam' :
            opt = Adam(self.w,self.lr)

        elif self.optimizer == 'SGD':
            opt = SGD(self.w,self.lr)

        elif self.optimizer == 'Nesterov':
            opt = Nesterov(self.w,self.lr)

        costs=[]
        iters=[]
        for epoch in range(1,self.epoches):
            batches = Dataset()
            for k,X_batch, y_batch in enumerate(batches.generate_batches()):
                cache, A = self.model_forward(X_batch)     #FORWARD PROPOGATIONS
                cost = self.cost_f(A, y_batch)                                  #COST FUNCTION
                grad = self.backward(X_batch, y_batch, cache) #BACKWARD PROPAGATION 
                if self.optimizer == 'adam':
                    opt.step(grad,k)
                else:
                    opt.step(grad)
                if k%5==0:
                    costs.append(cost)
                    iters.append(k)
                if k % 100 == 0:
                    print('cost of iteration______{}______{}'.format(k,cost))
        return losses, iters
            
    def plot_fn(itr,cost_momentum,cost_rms,cost_adam,cost_sgd):
        plt.plot(itr,cost_momentum,color="blue",label="mommentum")
        plt.plot(itr,cost_rms,color="black",label="rmsprop")
        plt.plot(itr,cost_adam,color="red",label="adam")
        plt.plot(itr,cost_sgd,color="green",label="minibatch-sgd")
        plt.xlabel('num_iter')
        plt.ylabel('cost')
        plt.legend()
        plt.title('visualization of different optimizers')
        plt.show()


