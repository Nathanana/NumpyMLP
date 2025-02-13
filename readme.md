# MNIST Mutlilayer Perceptron 

***Coded in NumPy***

I completed this project in order to learn more about basic neural networks. I've used PyTorch before, but so much is done already that I didn't fully understand the math behind the networks.
In this project I implemented a basic multi layer perceptron class that can train on any data, though it is limited to only using Linear Layers, as I haven't implemented any other kinds.
I added a Test.py file that can then take a saved model and find it's accuracy across the testing data, then displaying some of the predictions, along with 25 incorrectly predicted images.

## Components

- `nn`
  - Contains the classes for the MLP, along with some code to get started training one
- `models`
  - `MNIST_MLP_1 | Test Accuracy: 97.60%` Two hidden layers with 128 neurons each 
  - `MNIST_MLP_2 | Test Accuracy: 93.93%` Two hidden layers with 10 neurons each 
  - `MNIST_MLP_3 | Test Accuracy: 97.62%` Two hidden layers with 512 neurons each 
  - `MNIST_MLP_4 | Test Accuracy: 95.66%` Three hidden layers with 20 neurons each 
  - `MNIST_MLP_BW | Test Accuracy: 92.82%` Same as model 2, but trained on only black and white data, no grayscale, to see if it would perform better on the DrawingApp, though no noticeable difference arose
- `Test.py`
  - Contains code to test out a trained and saved MLP
- `Main.py`
  - Contains the DrawingApp class, which creates a drawing interface to see how the MLP can guess your own drawn numbers
  - Much less accurate than on testing data, likely since the MNIST dataset is hand drawn and the app can only allow mouse drawn numbers
 
## Takeaways

Increasing layers is more efficient to train than increasing neurons, though the added complexity may end up being prone to overfitting. I'll have to do more testing on that. 
Going from two layers of 128 neurons to 4x that at 512 gave just a .02% improvement. This is always random as different starting parameters may have yielded different results, but it's clear
that we approach diminishing returns fairly quickly. Model 2 took just a couple of minutes to train while Model 3 took 2 hours. There is much to be optimized, especially since batch processing hasn't been implemented,
but this was definitely more of an learning project than a full production.

The app was meant to help visualize, though there must be enough differences between that implementation and the actual training data that it's accuracy ends up closer to 60%, It's still cool to see, though.
It works by taking the drawn image and then scaling it all the way down to the same resolution as the data.

## Math Summary

The MLP takes the image in as a vector of pixel values. This can be imagined as a 28x28 image turning into one long 784 length vector. Once the MLP has the vector, each neuron in the next layer multiplies every
neuron in the previous layer by a specific weight and then adds a bias. This is done until an output is found. The output is then fed through a cost function, which calculates how wrong the model was. 

To put it short, the cost is like a ball up on a hill with lots of potential energy. We're looking for the gradient, or direction which would bring it higher, increasing it's potential energy the most out of all other directions,
and then moving it in the opposite direction. The position on the hill, x and y, are like the weights, and the height or the potential energy is like the cost.

One way of looking at the cost function is as a function of the actual output and an expected output, but another is by looking at it as a function of the weights and biases, the input, and the expected output. In this way, we have
thousands of input variables, and we can now find the rate of change of the cost function with respect to each one. This is the same as finding the gradient of a function like $f(x,y)$ which is done via $\triangledown \cdot f(x, y)$.
Our function is much more complex, but the same in principle. To then maximize the decrease in our cost function, we subtract the gradient (typically multiplied by a small coefficient) from every weight and bias, then run through
the process again. 

To find $\frac{\partial C}{\partial w_i}$ where C is the cost and $w_i$ is some arbitrary weight, the chain rule is used. $\frac{\partial C}{\partial w_i} = \frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$
where a is the activation of the last layers (the models prediction), z is the linear combination of inputs in the previous layer of the network, and w is the weights of the neuron. This is not an easy concept, and there are much
better resources than I to explain it more in depth, so this is where I'll leave it.
