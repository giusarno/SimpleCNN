import torch
class Dropout:

    p = None

    def __init__(self,p):
        self.p = p
    def getLayerName(self,count):
        ln = "dropout_%i" % count
        return ln
    def getTorchLayer(self,dimension,input_channels):
        self.out_channels=input_channels

        if dimension == 1:
            return torch.nn.Dropout(self.p)
        elif dimension == 2:
            return torch.nn.Dropout2d(self.p)
        elif dimension == 3:
            return torch.nn.Dropout3d(self.p)
        else:
            raise Exception("dimension out of range. Dimension = %i" % dimension)

    def print(self):
        print("Print Dropout")
        print( self.p)
    def get_output_size(self,inputsize):
        return inputsize



