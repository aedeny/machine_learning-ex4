import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import datasets


class NeuralNetwork(nn.Module):
    def __init__(self, image_size, h1_size, h2_size, mnist_out_size, batch=False, dropout=False, convolution=False):
        super(NeuralNetwork, self).__init__()
        self.image_size = image_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.mnist_out_size = mnist_out_size
        self.batch = batch
        self.dropout = dropout
        self.convolution = convolution

        if convolution:
            self.convolution1 = nn.Conv2d(1, 10, kernel_size=5)
            self.convolution2 = nn.Conv2d(10, 20, kernel_size=5)

            if dropout:
                self.convolution2_dropout = nn.Dropout2d()

            self.fc0 = nn.Linear(320, self.h1_size)
        else:
            self.fc0 = nn.Linear(self.image_size, self.h1_size)

        if batch:
            self.fc0_bn = nn.BatchNorm1d(self.h1_size)
        self.fc1 = nn.Linear(self.h1_size, self.h2_size)

        if batch:
            self.fc1_bn = nn.BatchNorm1d(self.h2_size)
        self.fc2 = nn.Linear(self.h2_size, self.mnist_out_size)

        if batch:
            self.fc2_bn = nn.BatchNorm1d(self.mnist_out_size)

    def forward(self, x):
        if self.convolution:
            x = f.relu(f.max_pool2d(self.convolution1(x), 2))
            x = f.relu(f.max_pool2d(self.convolution2_dropout(self.convolution2(x)), 2))

        if self.convolution:
            x = x.view(-1, 320)
        else:
            x = x.view(-1, self.image_size)

        x = f.relu(self.fc0_bn(self.fc0(x)))
        x = f.relu(self.fc1_bn(self.fc1(x)))

        if self.dropout:
            x = f.dropout(x, 0.2, self.training)
        x = f.relu(self.fc2_bn(self.fc2(x)))

        if self.dropout:
            x = f.dropout(x, 0.2, self.training)

        return f.log_softmax(x, dim=1)


def train(epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


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
        train(epoch, model)
        test()
