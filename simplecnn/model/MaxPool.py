import torch

class MaxPool:

    kernel_size = None
    stride = None
    padding = None

    def __init__(self,kernel_size, stride, padding):
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.out_channels=0

    def getLayerName(self,count):
        ln = "maxpool_%i" % count
        return ln

    def getTorchLayer(self,dimension,input_channels):
        self.out_channels=input_channels

        if dimension == 1:
            return torch.nn.MaxPool1d(self.kernel_size,self.stride,self.padding)
        elif dimension == 2:
            return torch.nn.MaxPool2d(self.kernel_size,self.stride,self.padding)
        elif dimension == 3:
            return torch.nn.MaxPool3d(self.kernel_size,self.stride,self.padding)
        else:
            raise Exception("dimension out of range. Dimension = %i" % dimension)

    def print(self):
        print("Print MaxPool")
        print( self.kernel_size, self.stride, self.padding)

    def get_output_size(self,inputsize):
        return self.calcOutputSize(inputsize,[self.kernel_size],[self.stride],[self.padding])

    def calcScalarLayerSize(self,input,kernel,stride,padding):
        output = int(((input+2*padding-(kernel-1)-1)/stride)+1)
        return output

    def calcOutputSize(self,input,kernel,stride,padding):
        inputlen = len(input)
        kernellen = len(kernel)
        stridelen = len(stride)
        paddinglen = len(padding)

        if kernellen == 1:
            kernel = [kernel[0]] * inputlen
            kernellen = inputlen
        if stridelen == 1:
            stride = [stride[0]] * inputlen
            stridelen = inputlen
        if paddinglen == 1:
            padding = [padding[0]] * inputlen
            paddinglen = inputlen

        if inputlen == kernellen == stridelen == paddinglen:
            output=[]
            for i in range(inputlen):
                output.append(self.calcScalarLayerSize(input[i],kernel[i],stride[i],padding[i]))
                #print(output[i])
            return output
        else:
            raise Exception("input,kernel,stride,padding dimension is not the same input %s,kernel %s,stride %s,padding %s" % (inputlen,kernellen,stridelen,paddinglen))





