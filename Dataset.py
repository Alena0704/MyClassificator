import numpy as np
import h5py
class Dataset():
    def __init__(self, batch_size=100, normalize=False):
        self.batch_size = batch_size
        hf1 = h5py.File('/pythonfile/popular_optimizer/train_datasets.h5','r')  #READ THE DATASETS
        self.X_train = hf1.get('X_train')     #GET THE X TRAINING DATASETS
        self.Y_train = hf1.get('Y_train')
        if normalize:
            self.normilize()
        print("Shape of X_train {}  and  Y_train {}".format(self.X_train.shape,self.Y_train.shape))

    def get_weights(self):
        n, k = self.X_train.shape        
        np.random.seed(42)
        # Вектор столбец в качестве весов
        return np.random.randn(k + 1)

    def __normilize_medium(self,X):
        a = X.mean()
        std = X.std()
        return sum(X-a)/std

    def normilize(self):
        self.X_train = np.reshape(self.X_train,(self.X_train.shape[0],-1))  #FLATTEN   (209, 64 *64 * 3)
        self.X_train = self.__normilize_medium(self.X_train)           #NORMALIZING THE DATASETS
        self.Y_train = np.reshape(self.Y_train,(self.Y_train.shape[0],1))                # MAKING SURE IT IS IN CORRECT SHAPE

    def generate_batches(self):
        m = self.X_train.shape[0]  
        miniBatches = [] 
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = self.X_train[permutation, :]
        shuffled_Y = self.Y_train[permutation, :].reshape((m,1))   #sure for uptpur shape

        num_minibatches = m // self.batch_size
        for k in range(0, num_minibatches):
            miniBatch_X = shuffled_X[k * self.batch_size:(k + 1) * self.batch_size,:]
            miniBatch_Y = shuffled_Y[k * self.batch_size:(k + 1) * self.batch_size,:]
            miniBatch = (miniBatch_X, miniBatch_Y)
            yield miniBatch
        