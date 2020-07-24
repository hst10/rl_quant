import sys
from itertools import product

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time

from pytorch_mvm_class import *


class lenet_mnist_mvm(nn.Module):
    def __init__(self, quant_cfg):
        super(lenet_mnist_mvm, self).__init__()

        assert(len(quant_cfg) == 4)

        qc = quant_cfg[0]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.conv1 = Conv2d_mvm(1, 6, 5, padding=2,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

        self.pool1 = nn.MaxPool2d(2, 2)

        qc = quant_cfg[1]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.conv2 = Conv2d_mvm(6, 16, 5, padding=2,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

        self.pool2 = nn.MaxPool2d(2, 2)

        qc = quant_cfg[2]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.conv3 = Conv2d_mvm(16, 32, 5, padding=2,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

        self.pool3 = nn.MaxPool2d(2, 2)

        qc = quant_cfg[3]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.fc1 = Linear_mvm(32 * 4 * 4, 10,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x


class lenet_cifar10_mvm(nn.Module):
    def __init__(self, quant_cfg):
        super(lenet_cifar10_mvm, self).__init__()

        assert(len(quant_cfg) == 4)

        qc = quant_cfg[0]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.conv1 = Conv2d_mvm(3, 6, 5, padding=2,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

        self.pool1 = nn.MaxPool2d(2, 2)

        qc = quant_cfg[1]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.conv2 = Conv2d_mvm(6, 16, 5, padding=2,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

        self.pool2 = nn.MaxPool2d(2, 2)

        qc = quant_cfg[2]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.conv3 = Conv2d_mvm(16, 32, 5, padding=2,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

        self.pool3 = nn.MaxPool2d(2, 2)

        qc = quant_cfg[3]
        weight_bits, weight_bit_frac, input_bits, input_bit_frac, \
        adc_bit, acm_bits, acm_bit_frac = qc

        self.fc1 = Linear_mvm(32 * 4 * 4, 10,
            bit_slice=2, bit_stream=1,
            weight_bits=weight_bits, weight_bit_frac=weight_bit_frac,
            input_bits=input_bits, input_bit_frac=input_bit_frac,
            adc_bit=adc_bit, acm_bits=acm_bits, acm_bit_frac=acm_bit_frac)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x


class lenet_mnist_mvm_adc(nn.Module):
    def __init__(self, adc_bit):
        super(lenet_mnist_mvm_adc, self).__init__()
        self.conv1 = Conv2d_mvm(1, 6, 5, padding=2, adc_bit=adc_bit[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2d_mvm(6, 16, 5, padding=2, adc_bit=adc_bit[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2d_mvm(16, 32, 5, padding=2, adc_bit=adc_bit[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = Linear_mvm(32 * 4 * 4, 10, adc_bit=adc_bit[3])

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x


class lenet_cifar10_mvm_adc(nn.Module):
    def __init__(self, adc_bit):
        super(lenet_cifar10_mvm_adc, self).__init__()
        self.conv1 = Conv2d_mvm(3, 6, 5, padding=2, adc_bit=adc_bit[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2d_mvm(6, 16, 5, padding=2, adc_bit=adc_bit[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2d_mvm(16, 32, 5, padding=2, adc_bit=adc_bit[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = Linear_mvm(32 * 4 * 4, 10, adc_bit=adc_bit[3])

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x


class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        return x

def prepare_mnist(batch_size=4):
    transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize(32),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                          download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                         download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4)

    return trainloader, testloader

def prepare_cifar10(batch_size=4):
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def train(net, trainloader, num_epoch, gpu):
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if gpu:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1500 == 1499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def save_model(net, path):
    torch.save(net.state_dict(), path)

def load_model(net, path):
    net.load_state_dict(torch.load(path))
    return net

def load_model_manual(net, net_mvm, path):
    net.load_state_dict(torch.load(path))

    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            weights_conv.append(m.weight.data.clone())
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            bn_data.append(m.weight.data.clone())
            bn_bias.append(m.bias.data.clone())
            running_mean.append(m.running_mean.data.clone())
            running_var.append(m.running_var.data.clone())
            num_batches.append(m.num_batches_tracked.clone())
        elif isinstance(m, nn.Linear):
            weights_lin.append(m.weight.data.clone())

    i = 0
    j = 0
    k = 0

    # print("bn_data size = ", len(bn_data))
    for m in net_mvm.modules():
        if isinstance(m, Conv2d_mvm):
            m.weight.data = weights_conv[i]
            i = i + 1
        #print(m.weight.data)
        #raw_input()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j + 1
        elif isinstance(m, Linear_mvm):
            m.weight.data = weights_lin[k]
            k = k + 1
    # model.cuda()
    # model_mvm.cuda()



def evaluate(net, testloader, gpu):
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # print(total)
            if total > 100: 
                break
            if gpu:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images: %.2f %%' % 
    #                 (100 * correct / float(total)))

    return (correct / float(total))


def train_lenet():
    # trainloader, testloader = prepare_mnist(batch_size=512)
    trainloader, testloader = prepare_cifar10(batch_size=512)
    # net = lenet_mnist_mvm([4,4,4,4])
    net = Net_cifar10()
    train(net, trainloader, 10, gpu=True)
    save_model(net, 'tmp_cifar_net_mvm.pth')


def exhaust_adc():
    if len(sys.argv) < 2:
        exit(0)

    task_idx = int(sys.argv[1])
    print('task_idx = ', task_idx)

    # trainloader, testloader = prepare_mnist(batch_size=512)
    trainloader, testloader = prepare_cifar10(batch_size=512)
    # net = lenet_mnist_mvm([4,4,4,4])
    # train(net, trainloader, 2, gpu=True)
    # save_model(net, 'mnist_net_mvm.pth')

    bits = [1, 2, 4, 6, 8]
    tasks = list(product(bits, bits, bits, bits))
    tasks.sort(reverse=True, key=lambda x: sum(x))
    task_list = [ tasks[i::6] for i in range(6) ]

    task_queue = task_list[task_idx]

    print(task_queue)

    for task in task_queue:
        net = lenet_cifar10_mvm_adc(task)
        # net = Net_mnist()
        load_model(net, './models/cifar_net_66.pth')
        acc = evaluate(net, testloader, gpu=True)
        res = ", ".join([ str(val) for val in task ]) + ', ' + str(acc)
        print(res)
        with open('lenet_cifar10_mvm_' + str(task_idx) + '.csv', 'a') as fout:
            fout.write(res + "\n")

def evaluate_quant(quant_cfg, dataset):
    start_time = time.time()

    if dataset == 'cifar10':
        trainloader, testloader = prepare_cifar10(batch_size=512)
        net_mvm = lenet_cifar10_mvm(quant_cfg)
        # net     = Net_cifar10()
        load_model(net_mvm, './models/cifar_net_66.pth')
        # load_model_manual(net, net_mvm, './models/cifar_net_66.pth')
    elif dataset == 'mnist':
        trainloader, testloader = prepare_mnist(batch_size=512)
        net_mvm = lenet_mnist_mvm(quant_cfg)
        load_model(net_mvm, './models/mnist_net_98.92.pth')

    acc = evaluate(net_mvm, testloader, gpu=True)
    end_time = time.time()
    print(f"acc = {acc}")
    print(f"time = {end_time - start_time} s")

if __name__ == '__main__':
    # evaluate_quant(quant_cfg=[(32,24,32,24,10,32,24)]*4, dataset='mnist')
    # evaluate_quant(quant_cfg=[(32,16,32,16,10,32,16)]*4, dataset='mnist')
    evaluate_quant(quant_cfg=[(16,12,16,12,6,16,12)]*4, dataset='mnist')
    # evaluate_quant(quant_cfg=[(8,4,8,4,10,8,4)]*4, dataset='mnist')
