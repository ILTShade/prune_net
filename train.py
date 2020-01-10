#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as Data
import torchvision
import torchvision.transforms as Transforms

# args parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                    help='gpu index, -1 for cpu(default: -1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
# random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# dataset and data loader
def get_dataset(device):
    pin_memory = (device != torch.device('cpu'))
    mean_value = (0.49, 0.48, 0.44)
    std_value = (0.24, 0.23, 0.25)
    train_batch_size = 128
    test_batch_size = 100
    train_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(os.path.dirname(__file__), 'data'),
        train=True,
        transform=Transforms.Compose([
            Transforms.Pad(padding=4),
            Transforms.RandomCrop(32),
            Transforms.RandomHorizontalFlip(),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=mean_value, std=std_value),
        ]),
        download=True,
    )
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=pin_memory,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=os.path.join(os.path.dirname(__file__), 'data'),
        train=False,
        transform=Transforms.Compose([
            Transforms.ToTensor(),
            Transforms.Normalize(mean=mean_value, std=std_value),
        ]),
        download=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
    )
    print(f'MNIST dataset')
    print(f'train loader length is {len(train_loader)}')
    print(f'test loader length is {len(test_loader)}')
    return train_loader, test_loader

# net module
class My3x3Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(My3x3Conv, self).__init__()
        self.block_list = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block_list(x)
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.layer_list = nn.Sequential(
            My3x3Conv(3, 64),
            My3x3Conv(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            My3x3Conv(64, 128),
            My3x3Conv(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            My3x3Conv(128, 256),
            My3x3Conv(256, 256),
            My3x3Conv(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            My3x3Conv(256, 512),
            My3x3Conv(512, 512),
            My3x3Conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            My3x3Conv(512, 512),
            My3x3Conv(512, 512),
            My3x3Conv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.layer_list(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def get_net():
    net = VGG16(10)
    print(net)
    return net

# train and test
def train_net(train_loader, test_loader, net, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 60
    LR = 0.1
    MOMENTUM = 0.9
    MILESTONE = [20, 40]
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = lr_scheduler.MultiStepLR(optimizer, MILESTONE)
    print(f'EPOCHS: {EPOCHS}, LR: {LR}, MOMENTUM: {MOMENTUM}, MILESTONE: {MILESTONE}')
    for epoch in range(EPOCHS):
        net.train()
        scheduler.step()
        for batch_idx, (images, labels) in enumerate(train_loader):
            net.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch+1:3d}, {batch_idx:3d}|{len(train_loader):3d}, loss: {loss.item():2.4f}', end = '\r')
        eval_net(test_loader, net, device, epoch)
        torch.save(net.state_dict(), f'zoo/vgg16_origin.pth')
def eval_net(test_loader, net, device, epoch):
    net.to(device)
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    print('%s After epoch %d, accuracy is %2.4f' % (time.asctime(time.localtime(time.time())), epoch, test_correct / test_total))

if __name__ == '__main__':
    if args.gpu != -1:
        assert args.gpu in [0, 1, 2, 3]
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    train_loader, test_loader = get_dataset(device)
    net = get_net()
    train_net(train_loader, test_loader, net, device)
