import torch
class LogSoftMax:

    def getLayerName(self,count):
        ln = "logsoftmax_%i" % count
        return ln
    def print(self):
        print("Print LogSoftMax")
    def getTorchLayer(self,dimension,input_channels):
        self.out_channels=input_channels
        return torch.nn.LogSoftmax(dim=1)
    def get_output_size(self,inputsize):
        return inputsize

