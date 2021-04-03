import torch
import torchvision

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
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)


    net= themodel.Model()

    criterion = nn.NLLLoss()
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
