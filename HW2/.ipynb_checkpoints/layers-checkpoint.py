#coding=utf-8

# Library with layers for Technotrack task #1
import numpy as np
## Layes
class Linear:
    def __init__(self, input_size, output_size, no_b=False):
        '''
        Creates weights and biases for linear layer from N(0, 0.01).
        Dimention of inputs is *input_size*, of output: *output_size*.
        no_b=True - do not use interception in prediction and backward (y = w*X)
        '''
        #### YOUR CODE HERE
        #self.W = np.random.rand(input_size, output_size)
        self.W = np.random.rand(input_size, output_size)/100
        self.b = None
        if (no_b == False):
         #   self.b = np.random.rand(output_size)/100
            self.b = np.zeros(output_size)


    # N - batch_size
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        out = np.dot(X,self.W) + self.b
        self.fcache = (X, out)
        return out

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        #### YOUR CODE HERE
        X, out = self.fcache
        dLdx = np.dot(dLdy, self.W.T).reshape(X.shape)
        dLdw = np.dot(X.T, dLdy)
        self.bcache = dLdw
        return  dLdx
        
        
        

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        #### YOUR CODE HERE
        self.W = self.W - learning_rate * self.bcache
    
    def get_loss(self):
        pass
        


## Activations
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        out = 1 / (1 + np.exp(-X))
        self.X = X, out
        return out
        

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx.
        '''
        #### YOUR CODE HERE
        X, out = self.X
        sigm = 1 / (1 + np.exp(-X))
        dLdx = sigm * (1 - sigm)* dLdy
        return dLdx
        
    def step(self, learning_rate):
        pass
    

class ELU:
    def __init__(self, alpha):
        #### YOUR CODE HERE
        self.alpha = alpha

    def forward(self, X):
        #### YOUR CODE HERE
        self.X = X
        out = np.ones_like(X)
        #X_min = self.alpha * (np.exp(np.minimum(0, X)) - 1)
        #X_min[X > 0] = 0
        #X[X < 0] = 0
        #out = X + X_min
        out[X > 0] = X[X >0]
        out[X < 0] = self.alpha * (np.exp(X[X<0] - 1))
        return out

    def backward(self, dLdy):
        #### YOUR CODE HERE
        X = self.X
        dLdx = np.ones_like(self.X)
        dLdx[X > 0] = dLdy[X > 0]
        dLdx[X < 0] = self.alpha * np.exp(X[X < 0]) * dLdy[X < 0]
        
        #dLdx1 = np.ones_like(self.X)
        #dLdx1 = self.alpha * np.exp(self.X)
        #dLdx1[self.X > 0] = 1
        #dLdx1 *= dLdy
        return dLdx

    def step(self, learning_rate):
        pass


class ReLU:
    def __init__(self, a):
        #### YOUR CODE HERE
        self.a = a

    def forward(self, X):
        #### YOUR CODE HERE
        self.X = X
        #X_min = self.a * X
        #X[X < 0] = 0 
        #X_min[X > 0] = 0
        #out = X + X_min
        X[X < 0] *= self.a
        return X
      
    def backward(self, dLdy):
        #### YOUR CODE HERE
        X = self.X
        dLdx = np.ones_like(X)
        dLdx[X < 0] = self.a * dLdy[X < 0]
        dLdx[X > 0] = dLdy[X > 0]
        return dLdx

    def step(self, learning_rate):
        pass


class Tanh:
    def forward(self, X):
        #### YOUR CODE HERE
        tanh = 2/(1 + np.exp(-2*X)) - 1
        self.fcache = X, tanh
        return tanh

    def backward(self, dLdy):
        #### YOUR CODE HERE
        X, tanh = self.fcache
        dLdx = (1 - tanh**2)*dLdy
        return dLdx
        

    def step(self, learning_rate):
        pass


## Final layers, loss functions
class SoftMax_NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
        pass

    def forward(self, X):
        '''
        Returns SoftMax for all X (matrix with size X.shape, containing in lines probabilities of each class)
        '''
        #### YOUR CODE HERE
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.cache = X, probs
        return probs

    # y - true labels. Calculates dL/dy, returns dL/dX
    def backward(self, y):
        #### YOUR CODE HERE
        self.y = y
        X, probs = self.cache
        N, D = X.shape
        exp_scores = np.exp(X)
        dLdx = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        dLdx[range(N),y] -= 1
        dLdx /= N

        return dLdx
        
    def step(self, learning_rate):
        pass
    
    def get_loss(self):
        X, probs = self.cache
        N, D = X.shape
        y = self.y
        correct_logprobs = -np.log(probs[range(N),y] + 1e-8)
        loss = np.sum(correct_logprobs)/N
        return loss
    
    
    


class MSE_Error:
    # Saves X for backprop, X.shape = N x 1
    def forward(self, X):
        #### YOUR CODE HERE
        self.X = X
        return X

    # Returns dL/dy (y - true labels)
    def backward(self, y):
        #### YOUR CODE HERE
        X = self.X
        self.y = y.reshape(X.shape)
        return (X - self.y)
 
        
    def step(self, learning_rate):
        pass
    
    def get_loss(self):
        N, D = self.X.shape
        return np.sum((self.X - self.y)**2) / N
                
               


## Main class
# loss_function can be None - if the last layer is SoftMax_NLLLoss: it can produce dL/dy by itself
# Or, for example, loss_function can be MSE_Error()
class NeuralNetwork:
    def __init__(self, modules, loss_function=None):
        '''
        Constructs network with *modules* as its layers
        '''
        #### YOUR CODE HERE
        self.modules = modules
    
    def forward(self, X):
        #### YOUR CODE HERE
        #### Apply layers to input
        self.X = X
        modules = self.modules
        out = X
        for i in range(len(modules)):
            out = modules[i].forward(out)
        return out

    # y - true labels.
    # Calls backward() for each layer. dL/dy from k+1 layer should be passed to layer k
    # First dL/dy may be calculated directly in last layer (if loss_function=None) or by loss_function(y)
    def backward(self, y):
        #### YOUR CODE HERE
        modules = self.modules
        inp = y
        l = len(modules)
        self.y = y
        for i in range(l):
            inp = modules[l - i -1].backward(inp)
        return inp
        
    # calls step() for each layer
    def step(self, learning_rate):
        modules = self.modules
        lr = learning_rate
        for i in range(len(modules)):
            modules[i].step(lr)
        
    def get_loss(self):
       # print("Softmax loss: {}".format(self.modules[3].get_loss()))
       return self.modules[-1].get_loss()
    
        
    