import torch
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.metrics import confusion_matrix

def plot_arrays(xaxis,yaxis,legend,xlabel,ylabel):
    plt.plot(xaxis, yaxis, color='blue')
    plt.legend([legend], loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def fit_the_model(epochs,log_interval,net,optimizer,train_loader,test_loader,criterion,savedirectory=None):
    train_losses = []
    test_losses = []
    test_accuracy = []
    for epoch in range(1, epochs+1):
        max_acc=0
        train_l,train_c =train_with_metrics(epoch,log_interval,net,optimizer,train_loader,criterion)
        train_losses.append(train_l)
        loss,acc = test(net,test_loader,criterion)
        test_losses.append(loss)
        test_accuracy.append(acc)
        if epoch == 1:
            acc = train_l
            print("initialize acc = %f" % acc)
        if acc > max_acc:
            max_acc = acc
            print("new maximum accuracy = %f" % max_acc)
            if savedirectory != None:
                modelfilename = os.path.join(savedirectory,'model.pth')
                optimfilename = os.path.join(savedirectory,'optimizer.pth')
                print("savedirectory not null saving model.pth in %s" % modelfilename)
                print("savedirectory not null saving optimizer.pth in %s" % optimfilename)
                torch.save(net.state_dict(), modelfilename)
                torch.save(optimizer.state_dict(), optimfilename)

        print("Test loss/Acc %f/%f" % (loss, acc))
    return train_losses,test_accuracy,test_losses

def train_with_metrics(epoch,log_interval,network,optimizer,train_loader,loss_function):
    train_loss_x_epoch=0
    running_loss=0
    count_it=0
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        count_it +=1
        optimizer.zero_grad()
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss/avg: {:.6f}/{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),running_loss/count_it))

    train_loss_x_epoch=running_loss/count_it

    return train_loss_x_epoch,epoch

def train(epoch,log_interval,network,optimizer,train_loader,loss_function):
    train_losses=[]
    train_counter=[]
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            #print(epoch,batch_idx,len(data),len(train_loader),(batch_idx) + ((epoch-1)*len(train_loader)))

            train_counter.append((batch_idx) + ((epoch-1)*len(train_loader)))
            #print(train_counter)

    return train_losses,train_counter

def test(network,test_loader,loss_function):
    network.eval()
    test_losses=[]
    test_accuracy=[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += loss_function(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    #test_losses.append(test_loss)
    test_accuracy= 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))
    return test_loss,test_accuracy


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    for batch in loader:
        data, labels = batch

        preds = model(data)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels)
            ,dim=0
        )
    return all_preds,all_labels

def get_confusion_matrix(model,train_set,normalize=False):
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=10)
    train_preds,data = get_all_preds(model, train_loader)
    cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    return cm

def plot_confusion_matrix_for_train_set(net,train_set,normalize):

    cm = get_confusion_matrix(net,train_set,normalize)

    plt.figure(figsize=(50,50))
    plot_confusion_matrix(cm,classes=train_set.classes)
    plt.show()


def plot_confusion_matrix(cm,classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fmt = '.2f' if isinstance(cm[i, j], float) else 'd'
            #print(cm[i,j],fmt)
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

