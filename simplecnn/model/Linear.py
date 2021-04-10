import torch
class Linear:

    out_features = None

    def __init__(self,out_channels):
        self.out_channels=out_channels
    def getLayerName(self,count):
        ln = "linear_%i" % count
        return ln
    def print(self):
        print("Print Linear")
        print( self.out_features)
    def getTorchLayer(self,dimension,input_channels):
        return torch.nn.Linear(input_channels,self.out_channels)
    def get_output_size(self,inputsize):
        return inputsize


