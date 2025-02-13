import numpy as np
from lnn.NumpyMLP import MLP
import matplotlib.pyplot as plt
import tkinter as tk
from mnist import MNIST

MODEL = "MNIST_MLP_1"

mndata = MNIST("samples")

try:
    X_test, y_test = mndata.load_testing()
except Exception as e:
    print(f"Error loading MNIST data: {e}")
    exit(1)

model = MLP() 
model.load_model(f'models/np/{MODEL}.npz')

X_test = np.array(X_test) / 255.0
y_pred = model.predict(X_test)

for i in range(len(X_test)):
    correct = sum([y_p == y_t for y_p, y_t in zip(y_pred, y_test)])
    
correct_percentage = round(correct/len(X_test)*100, 2)

print(f"Test Accuracy: {correct_percentage}%")

j = np.random.randint(0, len(X_test) - 25)

# Shows 25 corectly predicted images
for i in range(j, j + 25):
    image = X_test[i].reshape(28, 28)
    plt.subplot(5, 5, i-j+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Pred: {y_pred[i]}')
    plt.axis('off')
    
plt.suptitle('25 Correctly Predicted Images', fontsize=16, y=0.98)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

mismatch = [i for i, (y_t, y_p) in enumerate(zip(y_test, y_pred)) if (y_t != y_p)]

# Shows 25 incorrectly predicted images
for i in range(min(25, len(mismatch))):
    image = X_test[mismatch[i]].reshape(28, 28)
    plt.subplot(5, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Pred: {y_pred[mismatch[i]]}')
    plt.axis('off')
    
plt.suptitle('25 Incorrectly Predicted Images', fontsize=16, y=0.98)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()