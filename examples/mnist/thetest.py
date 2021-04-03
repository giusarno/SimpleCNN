import torch
import torchvision

import matplotlib.pyplot as plt
import torch.nn as nn

import themodel
import simplecnn.cnnutil as cnnutil

if __name__ == '__main__':
    batch_size_train = 64
    batch_size_test = 1000

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_set=torchvision.datasets.MNIST('/files/', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

    #get the model and create the loss function
    net = themodel.Model()
    criterion = nn.NLLLoss(size_average=False)

    #Test with the model with no training.
    test_loss,test_accuracy = cnnutil.test(net,test_loader,criterion)

    #Test with the model as result of the training
    network_state_dict = torch.load("./results/model.pth")
    net.load_state_dict(network_state_dict)

    test_loss,test_accuracy = cnnutil.test(net,test_loader,criterion)

    conf_matrix = cnnutil.get_confusion_matrix(net,train_set,False)
    print (conf_matrix)

    plt.figure(figsize=(50,50))
    cnnutil.plot_confusion_matrix(conf_matrix,train_set.classes)
    plt.show()


