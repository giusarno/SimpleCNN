import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import themodel
import simplecnn.cnnutil as cnnutil
import matplotlib.animation as animation



def plotTrainLosses(train_counter,train_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

def fit_the_model(epochs,log_interval,net,optimizer,train_loader,test_loader,criterion):
    train_losses = []
    train_counter = []
    test_losses = []
    test_accuracy = []
    test_counter = []
    for epoch in range(1, epochs+1):
        max_acc=0
        train_l,train_c =cnnutil.train_with_metrics(epoch,log_interval,net,optimizer,train_loader,criterion)
        train_losses.append(train_l)
        train_counter.append(train_c)
        loss,acc = cnnutil.test(net,test_loader,criterion)
        test_losses.append(loss)
        test_accuracy.append(acc)
        test_counter.append(epoch)
        if epoch == 1:
            acc = train_l
            print("initialize acc = %f" % acc)
        if acc > max_acc:
            max_acc = acc
            print("new maximum accuracy = %f" % max_acc)
            torch.save(net.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

        print("Test loss/Acc %f/%f" % (loss, acc))
    return train_counter,train_losses,test_counter,test_accuracy,test_losses

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



    #test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    train_counter,train_losses,test_counter,test_accuracy,test_loss = cnnutil.fit_the_model(n_epochs,10,net,optimizer,train_loader,test_loader,criterion,'./results')
    #minloss=0
    #for epoch in range(1, n_epochs+1):
    ''' train_l, train_c =cnnutil.train(epoch,10,net,optimizer,train_loader,criterion)
        train_losses=train_losses+train_l
        train_counter=train_counter+train_c
        #print(train_losses)
        #print(train_counter)
        
        train_l,train_c =cnnutil.train_with_metrics(epoch,1000,net,optimizer,train_loader,criterion)
        train_losses.append(train_l)
        train_counter.append(train_c)
        if epoch == 1:
            minloss = train_l
            print("initialize minimum = %f" % minloss)
        if train_l < minloss:
            minloss = train_l
            print("new minimum = %f" % minloss)
            torch.save(net.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')
        loss,acc = cnnutil.test(net,test_loader,criterion)
        print("Test loss/Acc %f/%f" % (loss, acc))
'''

    cnnutil.plot_arrays(train_counter,train_losses,'loss','training examples','loss')
    cnnutil.plot_arrays(test_counter,test_accuracy,'accuracy','epoch','accuracy')
    #torch.save(net.state_dict(), './results/model.pth')
    #torch.save(optimizer.state_dict(), './results/optimizer.pth')
