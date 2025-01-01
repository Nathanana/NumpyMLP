from mnist import MNIST
import numpy as np
try:
    from LinearLayer import LinearLayer
except ImportError:
    from .LinearLayer import LinearLayer

class MLP:
    def __init__(self, n_layers, layer_dims):
        '''
        Args:  
            n_layers: (int) Number of layers (minus input layer)
            layer_dims: (list[int]) Size of each layers (Input, Hiddens..., Output)
        '''
        self.layers = []
        for i in range(n_layers):
            self.layers.append(LinearLayer(layer_dims[i], layer_dims[i+1]))
            
        self.layers[-1].activation_fn = lambda x: x
        self.layers[-1].activation_fn_grad = lambda x: x
        
        self.loss_fn = self.softmax_CE
        self.loss_grad = self.softmax_CE_grad
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
        return output_grad
    
    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)
    
    def softmax_CE(self, y_true, y_pred):
        softmax = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return -np.mean(np.log(softmax[np.arange(len(y_true)), y_true] + 1e-6))

    def softmax_CE_grad(self, y_true, y_pred):
        softmax = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        softmax[np.arange(len(y_true)), y_true] -= 1
        return softmax / y_true.shape[0]

 
    def train(self, X_train, y_train, learning_rate=1e-3, epochs=20):
        '''
        Args:
            X_train: (NDArray[float] self.input_size x size of training data) Input data
            y_train: (NDArray[float] self.input_size x 1) Expected results for each training data
            learning rate: Volatility, How much does each backprop change the weights
            epochs: Amount of training loops
        '''
        train_size=len(X_train)
        
        for epoch in range(1, epochs + 1): 
            
            running_loss = []     
            for sample in range(train_size):
                x = X_train[sample].reshape((1, -1))
                
                y_true_i = np.array([y_train[sample]])
                y_pred_i = self.forward(x)
                
                loss_grad = self.loss_grad(y_true_i, y_pred_i)
                
                self.backward(loss_grad)
                self.step(learning_rate)
                
                loss = self.loss_fn(y_true_i, y_pred_i)
                running_loss.append(loss)
                
            avg_loss = sum(running_loss)/len(running_loss)
            print(f"Epoch: {epoch}, Loss: {avg_loss}")
            
    def save_model(self, file_path):
        '''
        Saves the model's weights and biases to a .npz file.
        Args:
            file_path (str): Path to the file where model will be saved
        '''
        model_params = {}
        
        for i, layer in enumerate(self.layers):
            model_params[f'layer_{i}_weights'] = layer.weights
            model_params[f'layer_{i}_biases'] = layer.biases

        np.savez(file_path, **model_params)
        
    def load_model(self, file_path):
        '''
        Loads the model's weights and biases from a .npz file.
        Args:
            file_path (str): Path to the file from which the model will be loaded
        '''
        model_params = np.load(file_path)
        
        for i, layer in enumerate(self.layers):
            layer.weights = model_params[f'layer_{i}_weights']
            layer.biases = model_params[f'layer_{i}_biases']
            
    def predict(self, X_test):
        y_pred = []
        for sample in X_test:
            X_i = sample.reshape((1, -1))
            y_i = self.forward(X_i)
            # Softmax
            e_x = np.exp(y_i - np.max(y_i))
            y_pred.append(np.argmax(np.round(e_x / e_x.sum())))
        return y_pred
                

if __name__ == '__main__':
    from mnist import MNIST

    mndata = MNIST("samples")
    
    try:
        X_train, y_train = mndata.load_training()
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        exit(1)
        
    X_train = np.array(X_train) / 255.0

    model = MLP(n_layers=3, layer_dims=[784, 10, 10, 10])

    model.train(X_train, y_train)
    model.save_model("models/MNIST_MLP_")