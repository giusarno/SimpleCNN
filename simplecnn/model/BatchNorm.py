import torch
class BatchNorm:


    def getLayerName(self,count):
        ln = "batchnorm_%i" % count
        return ln
    def getTorchLayer(self,dimension,input_channels):
        self.out_channels=input_channels

        if dimension == 1:
            return torch.nn.BatchNorm1d(input_channels)
        elif dimension == 2:
            return torch.nn.BatchNorm2d(input_channels)
        elif dimension == 3:
            return torch.nn.BatchNorm3d(input_channels)
        else:
            raise Exception("dimension out of range. Dimension = %i" % dimension)

    def print(self):
        print("Print Dropout")
        print( self.p)
    def get_output_size(self,inputsize):
        return inputsize



