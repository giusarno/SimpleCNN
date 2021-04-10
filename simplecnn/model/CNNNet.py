import torch

class CNNNet(torch.nn.Module):

    def __init__(self, convolution,fullyconnected):
        super(CNNNet, self).__init__()

        self.convolution = convolution

        self.fullyconnected = fullyconnected

    def forward(self, x):
        for key,mod in self.convolution.items():
            x = mod(x)
        x = x.flatten(1) # flat
        #x = x.view(-1,320)

        for key,mod in self.fullyconnected.items():
            x = mod(x)
        return x