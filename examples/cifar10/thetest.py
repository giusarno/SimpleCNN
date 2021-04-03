import torch
import torchvision

import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms

import themodel
import simplecnn.cnnutil as cnnutil

if __name__ == '__main__':
    batch_size_train = 64
    batch_size_test = 1000

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='/cifar10data/', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/cifar10data/', train=False,
                                       download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    #get the model and create the loss function
    net = themodel.Model()
    criterion = nn.CrossEntropyLoss(size_average=False)

    #Test with the model with no training.
    test_loss,test_accuracy = cnnutil.test(net,test_loader,criterion)

    #Test with the model as result of the training
    network_state_dict = torch.load("./results/model.pth")
    net.load_state_dict(network_state_dict)

    test_loss,test_accuracy = cnnutil.test(net,test_loader,criterion)

    conf_matrix = cnnutil.get_confusion_matrix(net,testset,True)
    print (conf_matrix)

    plt.figure(figsize=(50,50))
    cnnutil.plot_confusion_matrix(conf_matrix,train_set.classes)
    plt.show()


