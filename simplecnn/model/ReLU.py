import torch
class ReLU:

    def print(self):
        print("Print ReLU")
    def getLayerName(self,count):
            ln = "relu_%i" % count
            return ln
    def getTorchLayer(self,dimension,input_channels):
        self.out_channels=input_channels
        return torch.nn.ReLU()

    def get_output_size(self,inputsize):
            return inputsize
