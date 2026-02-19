import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
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

def get_cnnmodel(device, filters):
    model = nn.Sequential(nn.Conv2d( 1,filters,kernel_size =3),
        nn.ReLU(),
        nn.MaxPool2d( 2),
        nn.Conv2d( filters, filters, kernel_size =3),
        nn.ReLU(),
        nn.MaxPool2d( 2),
        nn.Flatten() ,
        nn.Linear( filters * 5 * 5, 200),
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

def plot_results(filters, accuracy, xlabel, ylabel):
    plt.plot(filters, accuracy)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel}")
    plt.show()


if __name__ == "__main__":
    filters = [5, 10, 15, 20, 25]
    model_accuracy = []
    for filter in filters:
        print(f"Training CNN with {filter} filters")
        model = get_cnnmodel(device, filter)
        train_model(model, train_dl_cnn, n_epochs=5, device=device)
        epoch_accuracies = []
        for batch in test_dl_cnn:
            x, y = batch
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
        model_accuracy.append(np.mean(epoch_accuracies))
        print(f"accuracy for filter {filter} is {model_accuracy[-1]}")
    plot_results(filters, model_accuracy, "Number of Filters", "Test Accuracy")

    

