import math
import numpy as np
class SGD:
    def __init__(self, w, b, lr):
        self.w = w
        self.lr = lr
        self.b = b

    def step(self, grad):
        for l in range(len(self.w) // 2):
            self.w[l] = self.w[l] - self.lr * grad['dW' + str(l)]
            self.b[l] = self.b[l] - self.lr * grad['db' + str(l)]
        return self.w, self.b

class Momentum(SGD):
    def __init__(self, w, b, lr, a=0.9):
        super().__init__(w, lr)
        self.ht_w = []
        self.ht_b = []
        for i in range(len(w)//2 ):
            self.ht_w[i] = np.zeros(w[i].shape)
            self.ht_b[i] = np.zeros(b[i].shape)
        self.a=a

    
    def step(self, grads, w, b):
        for l in range(len(w) // 2 ):
            # HERE WE COMPUTING THE VELOCITIES 
            self.ht_w[l] = self.a * self.ht_w[l] + (1 - self.a) * grads['dW' + str(l)]
            self.ht_b[l] = self.a * self.ht_b[l] + (1 - self.a) * grads['db' + str(l)]
            #updating parameters W and b
            w[l] = w[l] - self.lr * self.ht_w[l]
            b[l] = b[l] - self.lr * self.ht_b[l]
        return w,b

class Nesterov(Momentum):
    def __init__(self, w, lr):
        super().__init__(w, lr)
    
    def step(self, grads, w, b):
        self.ht_w[t-1]
        #self.ht = self.a*self.ht - self.lr*grad
        #self.ht = self.a*self.ht+self.lr*(self.w-a*self.ht)
        pass

class RMS_Prop(SGD):
    def __init__(self, w, lr, eps = 0.1):
        super().__init__(w, lr)
        self.st_w = []
        self.st_b = []
        for i in range(len(w)//2 ):
            self.st_w[i] = np.zeros(w[i].shape)
            self.st_b[i] = np.zeros(b[i].shape)
        self.eps = eps

    def step(self, grads):
        for l in range(len(self.ht_w) // 2 ):
            # HERE WE COMPUTING THE VELOCITIES 
            self.st_w[l]= self.a * self.st_w + (1 - self.a) * np.square(grads['dW' + str(l)])
            self.st_b[l] = self.a * self.st_w[l] + (1 - self.a) * np.square(grads['db' + str(l)])
            
            #updating parameters W and b
            self.w[l] = self.w[l] - self.lr * grads['dW' + str(l)] / (np.sqrt( self.st_w[l] + self.eps))
            self.b[l] = self.b[l] - self.lr * grads['db' + str(l)] / (np.sqrt( self.st_b[l] + self.eps))

        return self.w, self.b
    
class Adam(Momentum, RMS_Prop):
    def __init__(self, w, lr, b1 = 0.9, b2 = 0.999):
        super().__init__(w, lr)
        self.b1 = b1
        self.b2 = b2

    def step(self, grads, t):
        epsilon = pow(10,-8)    # avoiding to divide to zero
        v_corrected = {}                         
        s_corrected = {} 
        # grads has the dw and db parameters from backprop
        # params  has the W and b parameters which we have to update 
        for l in range(len(self.w) // 2 ):
            # HERE WE COMPUTING THE VELOCITIES 

            self.ht_w[l] = self.beta1 * self.ht_w[l] + (1 - self.beta1) * grads['dW' + str(l)]
            self.ht_b[l] = self.beta1 * self.ht_b[l] + (1 - self.beta1) * grads['db' + str(l)]

            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(self.beta1, t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(self.beta1, t))


            self.st_w[l] = self.beta2 * self.st_w[l] + (1 - self.beta2) * np.power(grads['dW' + str(l)], 2)
            self.st_b[l] = self.beta2 * self.st_b[l] + (1 - self.beta2) * np.power(grads['db' + str(l)], 2)

            s_corrected["dW" + str(l)] = self.st_w[l]["dW" + str(l)] / (1 - np.power(self.beta2, t))
            s_corrected["db" + str(l)] = self.st_b[l]["db" + str(l)] / (1 - np.power(self.beta2, t))

            self.w[l] = self.w[l-1] - self.lr * v_corrected["dW" + str(l)] / np.sqrt(s_corrected["dW" + str(l)] + epsilon)
            self.b[l] = self.b[l-1] - self.lr * v_corrected["db" + str(l)] / np.sqrt(s_corrected["db" + str(l)] + epsilon)
        return self.w, self.b
