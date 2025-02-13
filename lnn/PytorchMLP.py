import torch
import torch.nn as nn
import torch.optim as optim
from mnist import MNIST
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class ptMLP(nn.Module):
    def __init__(self):
        super(ptMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    mndata = MNIST("samples")
    X_train, y_train = mndata.load_training()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    batch_size = 64 

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ptMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'models/pytorch/MNIST_MLP_PT1.pth')