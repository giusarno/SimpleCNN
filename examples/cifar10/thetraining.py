import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import themodel
import simplecnn.cnnutil as cnnutil


def plotTrainLosses(train_counter,train_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == '__main__':
    n_epochs = 5
    batch_size_train = 4
    learning_rate = 0.001
    momentum = 0.9
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='/cifar10data/', train=True,
                                        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)


    net= themodel.Model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


    train_losses = []
    train_counter = []
    test_losses = []

    #test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    for epoch in range(1, n_epochs+1):
        train_l, train_c =cnnutil.train(epoch,10,net,optimizer,train_loader,criterion)
        train_losses=train_losses+train_l
        train_counter=train_counter+train_c
        #print(train_losses)
        #print(train_counter)


    plotTrainLosses(train_counter,train_losses,)

    torch.save(net.state_dict(), './results/model.pth')
    torch.save(optimizer.state_dict(), './results/optimizer.pth')
