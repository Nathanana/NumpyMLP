import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        '''
        input_size: (int) Number of neurons feeding into layer
        output_size: (int) Number of neurons in layer
        '''
        self.weights = np.random.randn(input_size, output_size)/10.0
        self.biases = np.random.randn(output_size)/10.0
        self.activation_fn = self.ReLU
        
        self.weights_grads = np.empty_like(self.weights)
        self.biases_grads = np.empty_like(self.biases)
        self.activation_fn_grad = self.dReLU
    
    def forward(self, x):
        self.input = x
        self.linear_output = np.matmul(x, self.weights) + self.biases
        self.activation = self.activation_fn(self.linear_output)
        return self.activation
    
    def backward(self, output_grad):
        activation_grad = self.activation_fn_grad(output_grad)
        self.biases_grads = activation_grad.reshape(self.biases_grads.shape)
        self.weights_grads = np.matmul(self.input.T, activation_grad)
        return np.matmul(activation_grad, self.weights.T)
    
    def step(self, learning_rate=1e-3):
        self.weights -= self.weights_grads * learning_rate
        self.biases -= self.biases_grads * learning_rate
        
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def dReLU(self, output_grad):
        return np.where(self.linear_output > 0, output_grad, 0)
