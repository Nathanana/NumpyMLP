import torch
from lnn.PytorchMLP import ptMLP
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

MODEL = "MNIST_MLP_PT1"

mndata = MNIST("samples")

try:
    X_test, y_test = mndata.load_testing()
except Exception as e:
    print(f"Error loading MNIST data: {e}")
    exit(1)

# Convert to tensor
X_test = np.array(X_test, dtype=np.float32) / 255.0
y_test = np.array(y_test, dtype=np.int64)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Load model
model = ptMLP()
model.load_state_dict(torch.load(f'models/pytorch/{MODEL}.pth'))
model.eval()  # Set to evaluation mode

# Make predictions
with torch.no_grad():
    y_pred = torch.argmax(model(X_test), dim=1)

# Compute accuracy
correct = (y_pred == y_test).sum().item()
correct_percentage = np.round(correct / len(X_test) * 100, 2)
print(f"Test Accuracy: {correct_percentage}%")

# Visualize 25 correctly predicted images
correct_indices = [i for i in range(len(y_test)) if y_pred[i] == y_test[i]]
j = np.random.randint(0, len(correct_indices) - 25)
plt.figure(figsize=(10, 10))
for i in range(25):
    idx = correct_indices[j + i]
    image = X_test[idx].reshape(28, 28)
    plt.subplot(5, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Pred: {y_pred[idx].item()}')
    plt.axis('off')

plt.suptitle('25 Correctly Predicted Images', fontsize=16, y=0.98)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# Visualize 25 incorrectly predicted images
mismatch = [i for i in range(len(y_test)) if y_pred[i] != y_test[i]]
plt.figure(figsize=(10, 10))
for i in range(min(25, len(mismatch))):
    idx = mismatch[i]
    image = X_test[idx].reshape(28, 28)
    plt.subplot(5, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Pred: {y_pred[idx].item()}')
    plt.axis('off')

plt.suptitle('25 Incorrectly Predicted Images', fontsize=16, y=0.98)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
