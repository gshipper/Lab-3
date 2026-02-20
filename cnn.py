import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import time
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the FashionMNIST training and testing data
fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
x_train, y_train = fmnist_train.data, fmnist_train.targets
x_test, y_test = fmnist_test.data, fmnist_test.targets

class FMNISTDataset(Dataset):
    """Dataset for MLP"""
    def __init__(self, x, y):
        
        # Flatten images to 784 and normalize to [0, 1]
        x = x.view(-1, 28*28).float()/255
        self.x, self.y = x, y
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    def __len__(self):
        return len(self.x)

# MLP dataloaders
train_dataset = FMNISTDataset(x_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = FMNISTDataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)

class FMNISTDataset_CNN(Dataset):
    """Dataset for CNN - keeps 2D image structure"""
    def __init__(self, x, y):
        
        # Reshape to (N, 1, 28, 28) to add channel dimension and normalize to [0, 1]
        x = x.view(-1, 1, 28, 28)
        x = x.float()/255
        self.x, self.y = x, y
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    def __len__(self):
        return len(self.x)

# CNN dataloaders
train_dataset_cnn = FMNISTDataset_CNN(x_train, y_train)
train_dl_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
test_dataset_cnn = FMNISTDataset_CNN(x_test, y_test)
test_dl_cnn = DataLoader(test_dataset_cnn, batch_size=32, shuffle=True)

def get_mlp_model(device):
    """Returns a MLP model with one hidden layer of 1000 units"""
    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    return model

def get_conv_output_size(kernel_size):
    """
    Calculates the output size of the convolutional layers after applying two 
    convolutional layers with the changing kernel sizes.
    The input image size is assumed to be 28x28.
    Uses the formula: Output Size = (Input Size - Kernel Size) + 1
    Assumes pytorch defaults for padding and stride
    """
    # Apply formula for first conv layer (input size of 28)
    after_conv1 = (28 - kernel_size) + 1

    # Max pool with kernel size of 2 and stride of 2
    after_pool1 = (after_conv1 - 2) // 2 + 1

    # Apply formula for second conv layer
    after_conv2 = (after_pool1 - kernel_size) + 1

    # Max pool again
    after_pool2 = (after_conv2 - 2) // 2 + 1

    # After second max pool, we have 128 channels and dimensions of after_pool2 x after_pool2
    return 128 * after_pool2 * after_pool2

def get_cnn_model(device, kernel_size=3):
    """Returns a CNN model with two convolutional layers and two linear layers"""
    
    # Calculate the linear input size based on the kernel size
    linear_input_size = get_conv_output_size(kernel_size)
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=kernel_size), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=kernel_size), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(linear_input_size, 200), nn.ReLU(),
        nn.Linear(200, 10)
    ).to(device)
    return model

def train_batch(x, y, model, opt, loss_fn):
    """Performs a single training step on a batch of data"""
    model.train()
    opt.zero_grad()
    batch_loss = loss_fn(model(x), y)
    batch_loss.backward()
    opt.step()
    return batch_loss.detach().cpu().numpy()

@torch.no_grad()
def accuracy(x, y, model):
    """Computes the accuracy of the model on a batch of data"""
    model.eval()
    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum((argmaxes == y).float()) / len(y)
    return s.cpu().numpy()

def train_model(model, train_dl, n_epochs, device):
    """Trains the model for n_epochs and returns the losses and accuracies"""
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)
    losses, accuracies = [], []

    for epoch in range(n_epochs):
        print(f"Running epoch {epoch + 1} of {n_epochs}")
        epoch_losses, epoch_accuracies = [], []

        # Train on each batch
        for batch in train_dl:
            x, y = batch
            batch_loss = train_batch(x, y, model, opt, loss_fn)
            epoch_losses.append(batch_loss)
        epoch_loss = np.mean(epoch_losses)

        # Compute accuracy on each batch
        for batch in train_dl:
            x, y = batch
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
        epoch_accuracy = np.mean(epoch_accuracies)

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

    return losses, accuracies

def test_model(model, test_dl):
    """Computes the test accuracy of the model on the test set"""
    test_accs = []
    for batch in test_dl:
        x, y = batch
        batch_acc = accuracy(x, y, model)
        test_accs.append(batch_acc)
    return np.mean(test_accs)

def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model"""
    total = 0
    
    # Iterate over all parameters in the model and sum their sizes
    for p in model.parameters():
        total += p.numel()
    return total

def experiment_1(n_epochs=5):
    """Study 1: Compare MLP vs CNN on FashionMNIST with fixed number of layers"""

    # Train MLP
    mlp = get_mlp_model(device)
    print(f"MLP has {count_parameters(mlp)} parameters")
    start = time.time()
    mlp_losses, mlp_train_accs = train_model(mlp, train_dl, n_epochs, device)
    mlp_time = time.time() - start

    # Test accuracy on MLP
    mlp_test_acc = test_model(mlp, test_dl)
    print(f"MLP test accuracy: {mlp_test_acc:.4f}")

    # Train CNN
    cnn = get_cnn_model(device)
    print(f"CNN has {count_parameters(cnn)} parameters")
    start = time.time()
    cnn_losses, cnn_train_accs = train_model(cnn, train_dl_cnn, n_epochs, device)
    cnn_time = time.time() - start

    # Test accuracy on CNN
    cnn_test_acc = test_model(cnn, test_dl_cnn)
    print(f"CNN test accuracy: {cnn_test_acc:.4f}")

    print(f"MLP training time: {mlp_time:.2f}s")
    print(f"CNN training time: {cnn_time:.2f}s")

    # Plot results
    plt.figure(figsize=(13, 3))

    plt.subplot(121)
    plt.title('Training Loss value over epochs CNN vs MLP')
    plt.plot(np.arange(n_epochs) + 1, mlp_losses, label='MLP')
    plt.plot(np.arange(n_epochs) + 1, cnn_losses, label='CNN')
    plt.legend()

    plt.subplot(122)
    plt.title('Training Accuracy value over epochs CNN vs MLP')
    plt.plot(np.arange(n_epochs) + 1, mlp_train_accs, label='MLP')
    plt.plot(np.arange(n_epochs) + 1, cnn_train_accs, label='CNN')
    plt.legend()

    plt.savefig('study1_results.png')
    plt.show()

def experiment_2(n_epochs=5):
    """Study 2: Vary kernel size from 2, 3, 5, 7, 9 with fixed layers and filters"""
    kernel_sizes = [2, 3, 5, 7, 9]
    results = {}

    for ks in kernel_sizes:
        print(f"\nTraining CNN with kernel_size={ks}")
        cnn = get_cnn_model(device, kernel_size=ks)
        print(f"Parameters: {count_parameters(cnn):,}")

        # Train and time the model
        start = time.time()
        losses, train_accs = train_model(cnn, train_dl_cnn, n_epochs, device)
        training_time = time.time() - start

        # Test accuracy
        test_acc = test_model(cnn, test_dl_cnn)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {training_time:.2f}s")

        # Store results for plotting
        results[ks] = {
            'losses': losses,
            'train_accs': train_accs,
            'test_acc': test_acc,
            'time': training_time
        }

    # Plot results
    plt.figure(figsize=(13, 3))

    plt.subplot(121)
    plt.title('Training Loss value over epochs')
    for ks in kernel_sizes:
        plt.plot(np.arange(n_epochs) + 1, results[ks]['losses'], label=f'kernel={ks}')
    plt.legend()

    plt.subplot(122)
    plt.title('Training Accuracy value over epochs')
    for ks in kernel_sizes:
        plt.plot(np.arange(n_epochs) + 1, results[ks]['train_accs'], label=f'kernel={ks}')
    plt.legend()

    plt.savefig('study2_results.png')
    plt.show()

if __name__ == "__main__":
    #experiment_1()
    experiment_2()