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
