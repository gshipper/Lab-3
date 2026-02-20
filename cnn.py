import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import time
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fmnist_train = datasets.FashionMNIST('~/data/FMNIST', download=True, train=True)
fmnist_test = datasets.FashionMNIST('~/data/FMNIST', download=True, train=False)
x_train, y_train = fmnist_train.data, fmnist_train.targets
x_test, y_test = fmnist_test.data, fmnist_test.targets
class FMNISTDataset(Dataset):
    
    def __init__(self, x, y):
        x = x.view(-1,28*28).float()/255
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
    def __init__(self, x, y):
        x = x.view(-1, 1, 28, 28)
        x = x.float()/255
        self.x, self.y = x, y
    def __getitem__(self, ix):
        return self.x[ix].to(device),self.y[ix].to(device)
    def __len__(self):
        return len(self.x)


train_dataset_cnn = FMNISTDataset_CNN(x_train, y_train)
train_dl_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
test_dataset_cnn = FMNISTDataset_CNN(x_test, y_test)
test_dl_cnn = DataLoader(test_dataset_cnn, batch_size=32, shuffle=True)

def get_mlp_model(device):
    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    return model

def get_cnn_model(device):
    model = nn.Sequential(nn.Conv2d( 1,64,kernel_size =3),
        nn.ReLU(),
        nn.MaxPool2d( 2),
        nn.Conv2d( 64, 128, kernel_size =3),
        nn.ReLU(),
        nn.MaxPool2d( 2),
        nn.Flatten() ,
        nn.Linear( 3200, 200),
        nn.ReLU(),
        nn.Linear( 200, 10)
    ).to(device)
    return model

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    opt.zero_grad()
    batch_loss = loss_fn(model(x), y)
    batch_loss.backward()
    opt.step()
    return batch_loss.detach().cpu().numpy()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    argmaxes = prediction.argmax(dim=1)
    s = torch.sum((argmaxes == y).float()) / len(y)
    return s.cpu().numpy()

def train_model(model, train_dl, n_epochs, device):
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)
    losses, accuracies = [], []

    for epoch in range(n_epochs):
        print(f"Running epoch {epoch + 1} of {n_epochs}")
        epoch_losses, epoch_accuracies = [], []

        for batch in train_dl:
            x, y = batch
            batch_loss = train_batch(x, y, model, opt, loss_fn)
            epoch_losses.append(batch_loss)
        epoch_loss = np.mean(epoch_losses)

        for batch in train_dl:
            x, y = batch
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
        epoch_accuracy = np.mean(epoch_accuracies)

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

    return losses, accuracies

def test_model(model, test_dl):
    test_accs = []
    for batch in test_dl:
        x, y = batch
        batch_acc = accuracy(x, y, model)
        test_accs.append(batch_acc)
    return np.mean(test_accs)


def count_parameters(model):
        """
        Counts the number of parameters in a PyTorch model.
        """
        total = 0

        # Iterate over all parameters in the model and sum their sizes
        for p in model.parameters():
            total += p.numel()
        return total



def experiment_1(n_epochs=5):

    # train MLP
    mlp = get_mlp_model(device)
    print(f"MLP has {count_parameters(mlp)} parameters")
    start = time.time()
    mlp_losses, mlp_train_accs = train_model(mlp, train_dl, n_epochs, device)
    mlp_time = time.time() - start
    # test accuracy on MLP
    mlp_test_acc = test_model(mlp, test_dl)
    print(f"MLP test accuracy: {mlp_test_acc:.4f}")

    # train CNN
    cnn = get_cnn_model(device)
    print(f"CNN has {count_parameters(cnn)} parameters")
    start = time.time()
    cnn_losses, cnn_train_accs = train_model(cnn, train_dl_cnn, n_epochs, device)
    cnn_time = time.time() - start
    cnn_test_acc = test_model(cnn, test_dl_cnn)
    print(f"CNN test accuracy: {cnn_test_acc:.4f}")

    print(f"MLP training time: {mlp_time:.2f}s")
    print(f"CNN training time: {cnn_time:.2f}s")

    # plot results
    plt.figure(figsize=(13, 3))

    plt.subplot(121)
    plt.title('Training Loss value over epochs CNN vs MLP')
    plt.plot(np.arange(n_epochs) + 1, mlp_losses, label='MLP')
    plt.plot(np.arange(n_epochs) + 1, cnn_losses, label='CNN')
    plt.legend()

    plt.subplot(122)
    plt.title('Testing Accuracy value over epochs CNN vs MLP')
    plt.plot(np.arange(n_epochs) + 1, mlp_train_accs, label='MLP')
    plt.plot(np.arange(n_epochs) + 1, cnn_train_accs, label='CNN')
    plt.legend()

    plt.savefig('study1_results.png')
    plt.show()



if __name__ == "__main__":
    
    experiment_1()