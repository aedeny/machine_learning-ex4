import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import datasets


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
        x = f.dropout(x, 0.5)
        x = f.relu(self.fc2(x))
        x = f.dropout(x, 0.5)

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


def train(model):
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


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)

        # sum up batch loss
        test_loss += f.nll_loss(output, target, size_average=False).data[0]

        # get the index of the max log-probability
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./train_data', train=True, download=True, transform=transforms),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./test_data', train=False, transform=transforms, download=True),
        batch_size=64, shuffle=True)

    model = NeuralNetwork(image_size=28 * 28)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, 10 + 1):
        train(model)
        test()
