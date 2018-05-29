import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import datasets
import numpy as np


class BasicNeuralNetwork(nn.Module):
    def __init__(self, image_size, hidden_layer1_size, hidden_layer2_size, mnist_output_size):
        super(BasicNeuralNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, hidden_layer1_size)
        self.fc1 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc2 = nn.Linear(hidden_layer2_size, mnist_output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return f.log_softmax(x, dim=1)


class DropoutNeuralNetwork(nn.Module):
    def __init__(self, image_size, hidden_layer1_size, hidden_layer2_size, mnist_output_size):
        super(DropoutNeuralNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, hidden_layer1_size)
        self.fc1 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc2 = nn.Linear(hidden_layer2_size, mnist_output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = f.dropout(x, 0.2, self.training)
        x = f.relu(self.fc2(x))
        x = f.dropout(x, 0.2, self.training)

        return f.log_softmax(x, dim=1)


class BatchNormalizationNeuralNetwork(nn.Module):
    def __init__(self, image_size, hidden_layer1_size, hidden_layer2_size, mnist_output_size):
        super(BatchNormalizationNeuralNetwork, self).__init__()
        self.image_size = image_size

        self.fc0 = nn.Linear(image_size, hidden_layer1_size)
        self.fc0_bn = nn.BatchNorm1d(hidden_layer1_size)
        self.fc1 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_layer2_size)
        self.fc2 = nn.Linear(hidden_layer2_size, mnist_output_size)
        self.fc2_bn = nn.BatchNorm1d(mnist_output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0_bn(self.fc0(x)))
        x = f.relu(self.fc1_bn(self.fc1(x)))
        x = f.relu(self.fc2_bn(self.fc2(x)))
        return f.log_softmax(x, dim=1)


class CombinedNeuralNetwork(nn.Module):
    def __init__(self, image_size, hidden_layer1_size, hidden_layer2_size, mnist_output_size, batch_size=64):
        super(CombinedNeuralNetwork, self).__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc0 = nn.Linear(320, hidden_layer1_size)
        self.fc0_bn = nn.BatchNorm1d(hidden_layer1_size)
        self.fc1 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_layer2_size)
        self.fc2 = nn.Linear(hidden_layer2_size, mnist_output_size)
        self.fc2_bn = nn.BatchNorm1d(mnist_output_size)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = f.relu(self.fc0_bn(self.fc0(x)))
        x = f.relu(self.fc1_bn(self.fc1(x)))
        x = f.dropout(x, 0.3, self.training)
        x = f.relu(self.fc2_bn(self.fc2(x)))
        x = f.dropout(x, 0.3, self.training)
        return f.log_softmax(x, dim=1)


def train(epoch, model, train_loader, optimizer, batch_size):
    model.train()
    train_loss = 0
    correct_train = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        prediction = output.data.max(1, keepdim=True)[1]
        correct_train += prediction.eq(labels.data.view_as(prediction)).cpu().sum()

    train_loss /= len(train_loader)
    print('\nTrain Epoch: {}\tAccuracy {}/{} ({:.3f}%)\tAverage loss: {:.3f}'.format(
        epoch, correct_train, (len(train_loader) * batch_size),
        100. * float(correct_train) / (len(train_loader) * batch_size), train_loss))

    return train_loss


def validate(epoch, model, valid_loader, batch_size):
    model.eval()

    validation_loss = 0
    correct_valid = 0
    for data, label in valid_loader:
        output = model(data)
        validation_loss += f.nll_loss(output, label, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_valid += pred.eq(label.data.view_as(pred)).cpu().sum()

    validation_loss /= (len(valid_loader) * batch_size)
    print('Validation Epoch: {}\tAccuracy: {}/{} ({:.3f}%)\tAverage loss: {:.3f}'.format(
        epoch, correct_valid, (len(valid_loader) * batch_size),
        100. * float(correct_valid) / (len(valid_loader) * batch_size), validation_loss))

    return validation_loss


def test(learning_model, test_loader):
    learning_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = learning_model(data)

        # Sums up batch loss
        test_loss += f.nll_loss(output, target, size_average=False).item()

        # Gets the index of the max log-probability
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_data_loaders(batch_size, validation_ratio):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Loads data sets.
    train_data_set = datasets.FashionMNIST('./train_data', train=True, download=True, transform=t)
    test_data_set = datasets.FashionMNIST('./test_data', train=False, transform=t, download=True)

    # Splits train data set to the corresponding validation ratio.
    train_size = len(train_data_set)
    indices = list(range(train_size))
    split = int(validation_ratio * train_size)

    valid_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(valid_idx)

    train_ldr = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=train_sampler)

    validation_ldr = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=validation_sampler)
    test_ldr = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=True)

    return train_ldr, validation_ldr, test_ldr


def main():
    # Settings
    hidden1_size = 100
    hidden2_size = 50
    mnist_output_size = 10
    mnist_image_size = 28 * 28
    num_of_epochs = 10
    batch_size = 64
    learning_rate = 0.01

    train_loader, validation_loader, test_loader = get_data_loaders(batch_size, 0.2)

    model = CombinedNeuralNetwork(mnist_image_size, hidden1_size, hidden2_size, mnist_output_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_of_epochs):
        train_loss = train(epoch, model, train_loader, optimizer, batch_size)
        valid_loss = validate(epoch, model, validation_loader, batch_size)

    test(model, test_loader)


if __name__ == '__main__':
    main()
