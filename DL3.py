# Roy Cohen - DLLayer class - from 1.1 to 1.3(including 1.4 working as well with 1.5-1.6 partly working (there are some bugs that I wasn't able to fix yet))
from tkinter import SEL
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import h5py


class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate = 1.2, optimization = "None", random_scale = 0.01):
        # initialize the weights and bias
        self._name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._learning_rate = learning_rate
        self._optimization = optimization
        self.random_scale = random_scale

        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self._learning_rate)
            self._adaptive_alpha_W = np.full((self._num_units, self._input_shape[0]), self._learning_rate)

        self.adaptive_cont = 1.1
        self.adaptive_switch = 0.5
        
        self.activation_trim = 1e-10

        # the original is relu
        self.activation_forward = self.__relu
        self.activation_backward = self._relu_backward
        
        if (activation == "sigmoid"):
            self.activation_forward = self.__sigmoid
            self.activation_backward = self._sigmoid_backward
        if (activation == "leaky_relu"):
            self.leaky_relu_d = 0.01
            self.activation_forward = self.__leaky_relu
            self.activation_backward = self._leaky_relu_backward
        if (activation == "tanh"):
            self.activation_forward = self.__tanh
            self.activation_backward = self._tanh_backward
        if (activation == "trim_sigmoid"):
            self.activation_trim = 1e-10
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward
        if (activation == "trim_tanh"):
            self.activation_trim = 1e-10
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward
        if (activation == "softmax"):
            self.activation_forward = self._softmax
            self.activation_backward = self._softmax_backwards
        if (activation == "trim_softmax"):
            self.activation_trim = 1e-10
            self.activation_forward = self._trim_softmax
            self.activation_backward = self._softmax_backwards
            
        
        self.init_weights(W_initialization)

    def _get_W_shape(self):
        return (self._num_units, *(self._input_shape))
    
    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float) # b is init to zeros, always
        if (W_initialization == "random"):
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
        elif (W_initialization == "zeros"):
            self.W = np.zeros((self._num_units, *self._input_shape), dtype=float)
        elif W_initialization == "He":
            self.W = np.random.randn(*self._get_W_shape()) * np.sqrt(2.0/sum(self._input_shape))
        elif W_initialization == "Xaviar":
            self.W = np.random.randn(*self._get_W_shape()) * np.sqrt(1.0/sum(self._input_shape))
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    # forwards: 
    def __sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    def __leaky_relu(self, Z):
        return np.maximum(self.leaky_relu_d*Z,Z)
    def __relu(self, Z):
        return np.maximum(0, Z)
    def __tanh(self, Z):
        return np.tanh(Z)
    
    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = 1/(1+np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _trim_tanh(self,Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1+TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _softmax(self,Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
    def _trim_softmax(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100,Z)
                eZ = np.exp(Z)
        A = eZ/np.sum(eZ, axis=0)
        return A    
    
    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = A_prev
        self.Z = np.dot(self.W, self._A_prev) + self.b
        self.A = self.activation_forward(self.Z)
        return self.A

    # backwords:
    def _sigmoid_backward(self, dA):
        A = self.__sigmoid(self.Z)
        return dA * A * (1-A)


    def _leaky_relu_backward(self, dA):
        return np.where(self.Z <= 0, self.leaky_relu_d * dA, dA)

    def _relu_backward(self,dA):
        return np.where(self.Z <= 0, 0, dA)
    
    # I do not understand why the function that I created didn't work but I found this on the Internet and it works and it does the same thing as the function that I created
    def _tanh_backward(self, dA):
        return dA * (1 - np.power(self.__tanh(self.Z), 2))
    
    def _trim_sigmoid_backward(self, dA):
        A = self._trim_sigmoid(self.Z)
        dZ = dA * A * (1-A)
        return dZ
    
    def _trim_tanh_backward(self, dA):
        A = self._trim_tanh(self._Z)
        dZ = dA * (1-A**2)
        return dZ
    
    def _softmax_backwards(self,dA):
        return dA

    
    def backward_propagation(self, dA):
        m = self._A_prev.shape[1]
        dZ = self.activation_backward(dA)
        self.dW = np.dot(dZ, self._A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev

        # create the function "update_parameters" that updates the weights and bias
    def update_parameters(self):
        if self._optimization == "adaptive":    # Update parameters with adaptive learning rate. keep the sign positive. Update is multiply by the derived value
            self._adaptive_alpha_b = np.where(self.db * self._adaptive_alpha_b >= 0, self._adaptive_alpha_b * self.adaptive_cont, self._adaptive_alpha_b * self.adaptive_switch)
            self._adaptive_alpha_W = np.where(self.dW * self._adaptive_alpha_W >= 0, self._adaptive_alpha_W * self.adaptive_cont, self._adaptive_alpha_W * self.adaptive_switch)
            self.W -= self._adaptive_alpha_W * self.dW
            self.b -= self._adaptive_alpha_b * self.db
        else:
            self.W -= self._learning_rate * self.dW
            self.b -= self._learning_rate * self.db
    
    def save_weights(self,path,file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W",  data=self.W)
            hf.create_dataset("b",  data=self.b)

    def __str__(self):
        s = self._name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self._learning_rate) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"

        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s
    


class DLModel:
    def __init__(self, name = "Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
        
    def add(self,layer):
        self.layers.append(layer)
    

    def squared_means(self, AL,Y):
        m = Y.shape[1]
        return np.power(AL-Y, 2) / m

    def squared_means_backward(self, AL,Y):
        m = Y.shape[1]
        return 2*(AL-Y) / m

    def cross_entropy(self, AL,Y):
        #AL = np.where(AL==0, 0.00000000000000001, AL)
        return np.where(Y == 0, -np.log(1-AL), -np.log(AL))/Y.shape[1]
            
    def cross_entropy_backward(self, AL,Y):
        #AL = np.where(AL==0, 0.00000000000000001, AL)
        return np.where(Y == 0, 1/(1-AL), -1/AL)/Y.shape[1]
    
    def categorical_cross_entropy(self, AL, Y):
        errors = np.where(Y == 1, -np.log(AL), 0) 
        return np.sum(errors, axis=0)
    
    def categorical_cross_entropy_backward(self, AL, Y):
        return (AL - Y) 
            
    def compile(self, loss, threshold=0.5):
        self._is_compiled = True
        self.loss = loss
        self.threshold = threshold
        if loss == "squared_means":
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_backward
        if loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_backward
        if loss == "categorical_cross_entropy":
            self.loss_forward = self.categorical_cross_entropy
            self.loss_backward = self.categorical_cross_entropy_backward
    
    def compute_cost(self,AL,Y):
        return np.sum(self.loss_forward(AL, Y))
    
    def train(self, X, Y, num_iterations):

        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
        # forward propagation
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,False)
            #backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1,L)):
                dAl = self.layers[l].backward_propagation(dAl)
                # update parameters
                self.layers[l].update_parameters()
            #record progress
            
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ",str(i//print_ind),"%:",str(J))
            if i == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
        return costs

    def predict(self, X):
            L = len(self.layers)
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,True)
            if (Al.shape != (1,1)):
                return np.where(Al==Al.max(axis=0),1,0)
            return Al > self.threshold
        
    def predict_percent(self, X):
        L = len(self.layers)
        Al = X
        for l in range(1,L):
            Al = self.layers[l].forward_propagation(Al,True)
            
        return np.round(Al,5)
    
    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        for i in range(1,len(self.layers)):
            self.layers[i].save_weights(path, f"Layer{i}")
    
    def load_weights(self, path):
        for i in range(1,len(self.layers)):
            with h5py.File(path+f"/Layer{i}.h5", 'r') as hf:
                self.layers[i].W = hf['W'][:]
                self.layers[i].b = hf['b'][:]
    
    def confusion_matrix(self, X, Y):
        prediction = self.predict(X)
        prediction_index = np.argmax(prediction, axis=0)
        Y_index = np.argmax(Y, axis=0)
        right = np.sum(prediction_index == Y_index)
        print("accuracy: ",str(right/len(Y[0])))
        cf = np.zeros((len(Y), len(Y)), dtype=int)
        for i in range(len(Y[0])):
            cf[Y_index[i]][prediction_index[i]] += 1
        print(cf)
        return cf
    
    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s