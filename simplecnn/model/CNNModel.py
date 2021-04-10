from collections import OrderedDict
import torch

import simplecnn
from simplecnn.model.BatchNorm import BatchNorm
from simplecnn.model.CNNNet import CNNNet
from simplecnn.model.Conv import Conv
from simplecnn.model.Dropout import Dropout
from simplecnn.model.Linear import Linear
from simplecnn.model.LogSoftMax import LogSoftMax
from simplecnn.model.MaxPool import MaxPool
from simplecnn.model.ReLU import ReLU


class CNNModel:
    model = []
    realconvmodel =OrderedDict()
    realfcmodel =OrderedDict()
    startfc = True
    countconv =0
    countmaxpool = 0
    countrelu = 0
    countdropout = 0
    countlinear =0
    countlogsoftmax =0
    countbatchnorm = 0
    theconvnet = None
    thefcnet = None



    def __init__(self, input_channels, input_size):
        self.dimension = len(input_size)
        self.input_channels = input_channels
        self.input_size = list(input_size)

    def addConvLayer(self, layer):
        tempcount = 0
        if  isinstance(layer,Conv):
            self.countconv +=1
            tempcount = self.countconv
        elif    isinstance(layer,MaxPool):
            self.countmaxpool +=1
            tempcount = self.countmaxpool
        elif    isinstance(layer,ReLU):
            self.countrelu +=1
            tempcount = self.countrelu
        elif    isinstance(layer,Dropout):
            self.countdropout +=1
            tempcount = self.countdropout
        elif    isinstance(layer,BatchNorm):
            self.countbatchnorm +=1
            tempcount = self.countbatchnorm

        else:
            raise Exception("The Layer is not in (Conv,MaxPool,ReLU,Dropout,BatchNorm)")

        self.model.append(layer)
        ln = layer.getLayerName(tempcount)
        torchlayer = layer.getTorchLayer(self.dimension,self.input_channels)

        self.realconvmodel.update({ln:torchlayer})
        self.input_channels = layer.out_channels
        #print (self.input_channels)
        self.theconvnet = torch.nn.ModuleDict(self.realconvmodel)
        self.input_size = layer.get_output_size(self.input_size)
        #print(ln,self.input_size)

    def multiplyList(self,myList) :

        # Multiply elements one by one
        result = 1
        for x in myList:
            result = result * x
        return result

    def addFCLayer(self,layer):
        tempcount = 0
        if self.startfc:
            self.input_channels = self.input_channels * self.multiplyList(self.input_size)
            self.startfc = False
            self.dimension = 1

        if      isinstance(layer,Linear):
            self.countlinear +=1
            tempcount = self.countlinear
        elif    isinstance(layer,LogSoftMax):
            self.countlogsoftmax +=1
            tempcount = self.countlogsoftmax

        elif    isinstance(layer,ReLU):
            self.countrelu +=1
            tempcount = self.countrelu

        elif    isinstance(layer,Dropout):
            self.countdropout +=1
            tempcount = self.countdropout

        elif    isinstance(layer,BatchNorm):
            self.countbatchnorm +=1
            tempcount = self.countbatchnorm

        else:
            raise Exception("The Layer is not in (Linear,LogSoftMax,ReLU,Dropout,BatchNorm)")

        self.model.append(layer)
        ln = layer.getLayerName(tempcount)
        torchlayer = layer.getTorchLayer(self.dimension,self.input_channels)
        self.realfcmodel.update({ln:torchlayer})
        self.input_channels = layer.out_channels
        self.thefcnet = torch.nn.ModuleDict(self.realfcmodel)

    def print(self):
        print ("THE CONV NET")
        print (self.theconvnet)
        print ("THE FC NET")
        print (self.thefcnet)

    def get_net(self):

        return CNNNet(self.theconvnet,self.thefcnet)

'''
def Conv(out_channels, kernel_size, stride, padding):

def MaxPool(kernel_size, stride,padding):

def ReLU():

def Dropout(p):

def Linear(output):
def LogSoftmax(dim=1):
'''