import torch
import torchvision

import torch.optim as optim
import torch.nn as nn
import themodel
import simplecnn.cnnutil as cnnutil


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
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    net= themodel.Model()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


    train_losses,test_accuracy,test_loss = cnnutil.fit_the_model(
        n_epochs,
        100,
        net,
        optimizer,
        train_loader,
        test_loader,
        criterion,
        './results')

    counter = [i+1 for i in range(n_epochs)]

    cnnutil.plot_arrays(counter,train_losses,'loss','epoch','loss')
    cnnutil.plot_arrays(counter,test_accuracy,'accuracy','epoch','accuracy')

