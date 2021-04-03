import getopt
import pandas as pd
import sys
CONV="Conv"
RELU="ReLU"
DROPOUT="Dropout"
MAXPOOL="MaxPool"
LINEAR="Linear"
LOGSOFTMAX="LogSoftmax"


dimension=0
filename='themodel.py'
csvfile='model.csv'
batchsize=0
inputsize=[]
inputdimension=0
inputsizevalue=""
inputchannels=0
currentinputdims=""
currentinput=[]
convcount=0
maxpoolcount=0
dropoutcount=0
relucount=0
linearcount=0
logsoftmaxcount=0

sspaces="       "
modelstring="""    def __init__(self):
       super(Model, self).__init__()
"""
forwardstring="""    def forward(self, x):
"""
modelheading="""import torch

class Model(torch.nn.Module):
"""
modelfooter="return x"

def multiplyList(myList) :

    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def convertSizeIntoPytoch(size):
    string = "("
    for i in range(len(size)):
        string=string+str(size[i])
        if i <len(size)-1:
            string=string+","
    string=string+")"
    return string

def generateLayerStringConv(count,dimension,in_channels, out_channels, kernel_size, stride, padding):
    layer = "self.conv_%i = torch.nn.Conv%id(in_channels=%i, out_channels=%i, kernel_size=%s, stride=%s, padding=%s)" %(count,dimension,in_channels, out_channels, kernel_size, stride, padding)
    return layer
def generateForwardStringConv(count):
    layer = "x=self.conv_%i(x)" %(count)
    return layer

def generateLayerStringMaxpool(count,dimension,kernel_size, stride,padding):
    layer = "self.max_pool_%i = torch.nn.MaxPool%id(kernel_size=%s, stride=%s,padding=%s)" %(count,dimension,kernel_size, stride,padding)
    return layer
def generateForwardStringMaxpool(count):
    layer = "x=self.max_pool_%i(x)" %(count)
    return layer

def generateLayerStringDopout(count,p):
    layer = "self.dropout_%i = torch.nn.Dropout(p=%.1f)" %(count,p)
    return layer
def generateForwardStringDropout(count):
    layer = "x=self.dropout_%i(x)" %(count)
    return layer
def generateLayerStringLinear(count,input,output):
    layer = "self.linear_%i = torch.nn.Linear(%i,%i)" %(count,input,output)
    return layer
def generateForwardStringLinear(count,input):
    layer=""
    if count ==1:
        layer=layer+"x = x.view(-1,%i)\n" %(input) + sspaces
    layer = layer + "x=self.linear_%i(x)" %(count)
    return layer
def generateLayerStringRelu(count):
    layer = "self.relu_%i = torch.nn.ReLU()" %(count)
    return layer
def generateForwardStringRelu(count):
    layer = "x=self.relu_%i(x)" %(count)
    return layer
def generateLayerStringLogSoftmax(count):
    layer = "self.logsoftmax_%i = torch.nn.LogSoftmax(dim=1)" %(count)
    return layer
def generateForwardStringLogSoftmax(count):
    layer = "x=self.logsoftmax_%i(x)" %(count)
    return layer
def calcScalarLayerSize(input,kernel,stride,padding):
    output = int(((input+2*padding-(kernel-1)-1)/stride)+1)
    return output
def calcScalarLayerSizeMaxpool(input,kernel,stride,padding):
    output = int(((input+2*padding-(kernel-1)-1)/stride)+1)
    return output
def calcMultiLayerSize(input,kernel,stride,padding):
    inputlen = len(input)
    kernellen = len(kernel)
    stridelen = len(stride)
    paddinglen = len(padding)

    if inputlen == kernellen == stridelen == paddinglen:
       output=[]
       for i in range(inputlen):
           output.append(calcScalarLayerSize(input[i],kernel[i],stride[i],padding[i]))
           #print(output[i])
       return output
    else:
        raise Exception("input,kernel,stride,padding dimension is not the same input %s,kernel %s,stride %s,padding %s" % (inputlen,kernellen,stridelen,paddinglen))

def calcMultiLayerSizeMaxpool(input,kernel,stride,padding):
    inputlen = len(input)
    kernellen = len(kernel)
    stridelen = len(stride)
    paddinglen = len(padding)

    if inputlen == kernellen == stridelen == paddinglen:
        output=[]
        for i in range(inputlen):
            output.append(calcScalarLayerSizeMaxpool(input[i],kernel[i],stride[i],padding[i]))
            #print(output[i])
        return output
    else:
        raise Exception("input,kernel,stride,padding dimension is not the same input %s,kernel %s,stride %s,padding %s" % (inputlen,kernellen,stridelen,paddinglen))

def dimToArray(dim,controldim):
    dim=str(dim)
    dims = dim.split("x")
    dims = list(map(float, dims))
    dims = list(map(int, dims))
    l=len(dims)

    if controldim == None:
        if l == 1:
            dimint=[]
            dimint[0]=int(dims[0])
            return l,dimint
        return l,dims

    else:
        if l > controldim:
                raise Exception("the dimesion on %s is > input dim %s" % (dim,controldim))
        elif l == 1:
            dims = [int(float(dims[0])) for i in range(controldim)]
    return len(dims),dims

def main(argv):
    global dimension
    global filename
    global csvfile
    global batchsize
    global inputsize
    global inputdimension
    global inputsizevalue
    global modelstring
    global inputchannels
    global currentinputdims
    global forwardstring
    global convcount
    global relucount
    global dropoutcount
    global maxpoolcount
    global linearcount
    global logsoftmaxcount

    try:
        opts, args = getopt.getopt(argv,"hi:d:b:c:f:v:",["insize=","dimension=","batchsize=","channels=","filename","csvfile"])
    except getopt.GetoptError:
        print('model.py -i <inputsize> -d <dimension> -b <batchsize> -c <channels> -f <filename> -v <csvfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
             print("model.py -i <inputsize> -d <dimension> -b <batchsize> -c <channels> -f <filename> -v <csvfile>")
             sys.exit()
        elif opt in ("-i","--insize"):
            inputsizevalue = arg
        elif opt in ("-c","--channel"):
            inputchannels = int(arg)
        elif opt in ("-d","--dimension"):
            dimension = int(arg)
        elif opt in ("-b","--batchsize"):
            batchsize = arg
        elif opt in ("-f","--filename"):
            filename = arg
        elif opt in ("-v","--csvfile"):
            csvfile = arg
    inputdimension,input = dimToArray(inputsizevalue,None)
    model= pd.read_csv(csvfile,header=None)
    currentinput=input
    for i in range(len(model)) :
        layertype=model.iloc[i,0]
        currentinputdims=inputdimension
        if layertype==LINEAR:
            linearcount=linearcount+1
            output = int(model.iloc[i,1])
            nofelements = inputchannels * multiplyList(currentinput)
            modelstring=modelstring+sspaces+generateLayerStringLinear(linearcount,nofelements,output)+"\n"
            forwardstring=forwardstring+sspaces+generateForwardStringLinear(linearcount,nofelements)+"\n"
            currentinputdims=output
            currentinput=[1,output]
            inputchannels=1
        if layertype==RELU:
            relucount=relucount+1
            modelstring=modelstring+sspaces+generateLayerStringRelu(relucount)+"\n"
            forwardstring=forwardstring+sspaces+generateForwardStringRelu(relucount)+"\n"
        if layertype==LOGSOFTMAX:
            logsoftmaxcount=logsoftmaxcount+1
            modelstring=modelstring+sspaces+generateLayerStringLogSoftmax(logsoftmaxcount)+"\n"
            forwardstring=forwardstring+sspaces+generateForwardStringLogSoftmax(logsoftmaxcount)+"\n"
        if layertype==DROPOUT:
            dropoutcount=dropoutcount+1
            p = round(float(model.iloc[i,1]),1)
            modelstring=modelstring+sspaces+generateLayerStringDopout(dropoutcount,p)+"\n"
            forwardstring=forwardstring+sspaces+generateForwardStringDropout(dropoutcount)+"\n"
        if layertype==CONV:
            convcount=convcount+1
            output =int(model.iloc[i,1])
            kernel =model.iloc[i,2]
            stride =model.iloc[i,3]
            padding=model.iloc[i,4]
            kernell,kerneldim = dimToArray(kernel,currentinputdims)
            stridel,stridedim = dimToArray(stride,currentinputdims)
            paddingl,paddingdim = dimToArray(padding,currentinputdims)

            size = calcMultiLayerSize(currentinput,kerneldim,stridedim,paddingdim)
            modelstring=modelstring+sspaces+generateLayerStringConv(convcount,dimension,inputchannels, output, convertSizeIntoPytoch(kerneldim), convertSizeIntoPytoch(stridedim), convertSizeIntoPytoch(paddingdim))+"\n"
            forwardstring=forwardstring+sspaces+generateForwardStringConv(convcount)+"\n"
            inputchannels=output
            currentinputdims=len(size)
            currentinput=size

        if layertype==MAXPOOL:
            maxpoolcount=maxpoolcount+1
            kernel =model.iloc[i,1]
            stride =model.iloc[i,2]
            padding=model.iloc[i,3]
            kernell,kerneldim = dimToArray(kernel,currentinputdims)
            stridel,stridedim = dimToArray(stride,currentinputdims)
            paddingl,paddingdim = dimToArray(padding,currentinputdims)
            size = calcMultiLayerSizeMaxpool(currentinput,kerneldim,stridedim,paddingdim)

            modelstring=modelstring+sspaces+generateLayerStringMaxpool(maxpoolcount,dimension, convertSizeIntoPytoch(kerneldim), convertSizeIntoPytoch(stridedim), convertSizeIntoPytoch(paddingdim))+"\n"
            forwardstring=forwardstring+sspaces+generateForwardStringMaxpool(maxpoolcount)+"\n"
            currentinputdims=len(size)
            currentinput=size

def saveModelFile():
    f = open(filename, "w")
    f.write (modelheading)
    f.write (modelstring)
    f.write (forwardstring)
    f.write (sspaces+modelfooter)
    f.close()

if __name__ == "__main__":
    main(sys.argv[1:])
    print (modelheading)
    print (modelstring)
    print (forwardstring)
    print (sspaces+modelfooter)
    saveModelFile()