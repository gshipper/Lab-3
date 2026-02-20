import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import time
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

def train_model(model, train_dl, n_epochs=5, device=device):
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
    plt.savefig(f"{xlabel}_vs_{ylabel}.png")
    plt.show()

def get_cnn_bn(device, filters=25, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(1, filters, kernel_size=kernel_size),
        nn.BatchNorm2d(filters),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(filters, filters, kernel_size=kernel_size),
        nn.BatchNorm2d(filters),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(filters * 5 * 5, 200),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200, 10),
    ).to(device)

def get_cnn_dropout(device, filters=25, kernel_size=3, p=0.5):
    return nn.Sequential(
        nn.Conv2d(1, filters, kernel_size=kernel_size),
        nn.ReLU(),
        nn.Dropout2d(p),
        nn.MaxPool2d(2),

        nn.Conv2d(filters, filters, kernel_size=kernel_size),
        nn.ReLU(),
        nn.Dropout2d(p),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(filters * 5 * 5, 200),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(200, 10),
    ).to(device)


def experiment_4():
    base_model = get_cnnmodel(device, filters=25)
    bn_model = get_cnn_bn(device)
    dropout_model = get_cnn_dropout(device)
    start_time = time.time()
    base_losses, base_accs = train_model(base_model, train_dl_cnn)
    end_time = time.time()
    base_time = end_time - start_time
    start_time = time.time()
    bn_losses, bn_accs = train_model(bn_model, train_dl_cnn)
    end_time = time.time()
    bn_time = end_time - start_time
    start_time = time.time()
    dropout_losses, dropout_accs = train_model(dropout_model, train_dl_cnn)
    end_time = time.time()
    dropout_time = end_time - start_time

    test_acc_base = []
    test_acc_bn = []
    test_acc_dropout = []
    for batch in test_dl_cnn:
        x, y = batch
        test_acc_base.append(accuracy(x, y, base_model))
        test_acc_bn.append(accuracy(x, y, bn_model))
        test_acc_dropout.append(accuracy(x, y, dropout_model))
    accuracies = [np.mean(test_acc_base), np.mean(test_acc_bn), np.mean(test_acc_dropout)]

    plt.figure(figsize=(13, 3))
    plt.subplot(121)
    plt.title("Training Loss Across Models")
    plt.plot(np.arange(len(base_losses)) + 1, base_losses, label='Base CNN')
    plt.plot(np.arange(len(bn_losses)) + 1, bn_losses, label='Batch Norm CNN')
    plt.plot(np.arange(len(dropout_losses)) + 1, dropout_losses, label='Dropout CNN')
    plt.legend()

    plt.subplot(122)
    plt.title("Test Accuracy Across Models")
    plt.plot(np.arange(len(base_accs)) + 1, base_accs, label='Base CNN')
    plt.plot(np.arange(len(bn_accs)) + 1, bn_accs, label='Batch Norm CNN')
    plt.plot(np.arange(len(dropout_accs)) + 1, dropout_accs, label='Dropout CNN')
    plt.legend()
    plt.savefig('experiment_4_results.png')
    plt.show()

    labels = ['Base CNN', 'Batch Norm CNN', 'Dropout CNN']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, accuracies)
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    
    axes[1].bar(labels, [base_time, bn_time, dropout_time])
    axes[1].set_title("Training Time")
    axes[1].set_ylabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig('experiment_4_comparison.png')
    plt.show()




if __name__ == "__main__":
    """
    filters = [5, 10, 15, 20, 25]
    model_accuracy = []
    model_times = []
    for filter in filters:
        print(f"Training CNN with {filter} filters")
        model = get_cnnmodel(device, filter)
        start_time = time.time()
        train_model(model, train_dl_cnn)
        end_time = time.time()
        model_times.append(end_time - start_time)
        epoch_accuracies = []
        for batch in test_dl_cnn:
            x, y = batch
            batch_acc = accuracy(x, y, model)
            epoch_accuracies.append(batch_acc)
        model_accuracy.append(np.mean(epoch_accuracies))
        print(f"accuracy for filter {filter} is {model_accuracy[-1]}")
    plot_results(filters, model_accuracy, "Number of Filters", "Test Accuracy")
    plot_results(filters, model_times, "Number of Filters", "Training Time (seconds)")
    """
    experiment_4()

    

