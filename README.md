# SimpleCNN

## Utility functions to simplify dealing with cnn networks. Powered by Pytorch

## Install packages
change "C:/Users/GiuseppeSarno/IdeaProjects" with the root of your project.

```
set PYTHONPATH=%PYTHONPATH%;C:/Users/GiuseppeSarno/IdeaProjects/SimpleCNN/
cd C:\Users\GiuseppeSarno\IdeaProjects\SimpleCNN
pip install -r ../../requirements.txt
```
Known issues:
I hit this problem https://github.com/pytorch/pytorch/issues/23823 when installing "torch"
and fixed as on https://www.reddit.com/r/pytorch/comments/c6cllq/issue_installing_pytorch/ew27hih/?utm_source=share&utm_medium=web2x
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem = 1

## Run the MNIST example from console & windows:

### Run the model Generation

In this example we will be using the following parameters:
input(image) size   = 28x28
dimension           = 2 (we are dealing with 2 dimension input)
channels            =1  (only grayscale is used)
batch size          =10 (number of inputs will go through the model at the same time) 
filename            = themodel.py (output filename)
inputfile           = model.csv (input file with model definition)

```
> python3 ./simplecnn/modelgen.py -i 28x28 -d 2 -c 1 -b 10 -f ./examples/mnist/themodel.py -v ./examples/mnist/model.csv
```

### Run the Training
```
> cd ./examples/mnist
> python3 thetraining.py
```
Output:
```
Train Epoch: 1 [0/60000 (0%)]   Loss/avg: 2.395239/2.395239
Train Epoch: 1 [6400/60000 (11%)]       Loss/avg: 1.123712/1.907594
Train Epoch: 1 [12800/60000 (21%)]      Loss/avg: 0.777390/1.434527
Train Epoch: 1 [19200/60000 (32%)]      Loss/avg: 0.646094/1.191016
Train Epoch: 1 [25600/60000 (43%)]      Loss/avg: 0.704663/1.036754
Train Epoch: 1 [32000/60000 (53%)]      Loss/avg: 0.431299/0.927583
..............................................
..............................................
Train Epoch: 5 [25600/60000 (43%)]      Loss/avg: 0.143671/0.159320
Train Epoch: 5 [32000/60000 (53%)]      Loss/avg: 0.211940/0.160901
Train Epoch: 5 [38400/60000 (64%)]      Loss/avg: 0.092523/0.161049
Train Epoch: 5 [44800/60000 (75%)]      Loss/avg: 0.093523/0.160512
Train Epoch: 5 [51200/60000 (85%)]      Loss/avg: 0.243839/0.159639
Train Epoch: 5 [57600/60000 (96%)]      Loss/avg: 0.073853/0.158468

Test set: Avg. loss: 0.0001, Accuracy: 9824/10000 (98%)

new maximum accuracy = 98.239998
savedirectory not null saving model.pth in ./results\model.pth
savedirectory not null saving optimizer.pth in ./results\optimizer.pth
Test loss/Acc 0.000066/98.239998
```
the following pictures are displayed:


![Loss Function](https://github.com/giusarno/SimpleCNN/blob/master/examples/mnist/loss.png)

### Run the test and validation:

`> python3 thetest.py`

Output:
```
Test set: Avg. loss: 2.3316, Accuracy: 1137/10000 (11%)


Test set: Avg. loss: 0.0657, Accuracy: 9834/10000 (98%)

Confusion matrix, without normalization

[[5885    2    5    0    2    3   16    1    5    4]
[   0 6688   34    1    4    0    2    8    5    0]
[   8   18 5855   12    8    0    3   37   16    1]
[   6    3   56 5983    0   31    1   19   18   14]
[   3   19    6    0 5765    0   13    5    4   27]
[   9    5    3   16    3 5333   20    3   25    4]
[  16    8    3    0   17   18 5842    0   14    0]
[   2   17   35    2   13    1    0 6172    1   22]
[  11   38   32   14   15   20   12    7 5666   36]
[  21   13    0   12   82   16    1   52   15 5737]]

```
the following picture is displayed:

![confusion matrix](https://github.com/giusarno/SimpleCNN/blob/master/examples/mnist/conf_matrix.png)


## Run the CIFAR10 example

### Run the model Generation
```
In this example we will be using the following parameters:
input(image) size   = 32x32
dimension           = 2 (we are dealing with 2 dimension input)
channels            = 3  (RGB is used)
batch size          = 4 (number of inputs will go through the model at the same time)
filename            = themodel.py (output filename)
inputfile           = model.csv (input file with model definition)
```

`> python3 ./simplecnn/modelgen.py -i 32x32 -d 2 -c 3 -b 4 -f ./examples/cifar10/themodel.py -v ./examples/cifar10/model.csv`

## Run the training

```
> cd ./examples/cifar10
> python3 thetraining.py
```
Output:
```
Files already downloaded and verified
Train Epoch: 1 [0/50000 (0%)]   Loss: 2.326186
Train Epoch: 1 [40/50000 (0%)]  Loss: 2.304344
..........................................
..........................................
Train Epoch: 5 [49800/50000 (100%)]     Loss: 1.370805
Train Epoch: 5 [49840/50000 (100%)]     Loss: 0.616962
Train Epoch: 5 [49880/50000 (100%)]     Loss: 0.784681
Train Epoch: 5 [49920/50000 (100%)]     Loss: 1.301057
Train Epoch: 5 [49960/50000 (100%)]     Loss: 1.799423
```

the following picture is displaied:

![training loss](examples/cifar10/loss.png)


### Run the test and validation:

`> python3 thetest.py`

Output:
```
Test set: Avg. loss: 2.3037, Accuracy: 943/10000 (9%)


Test set: Avg. loss: 1.1305, Accuracy: 6029/10000 (60%)

Normalized confusion matrix
[[0.647 0.033 0.033 0.037 0.017 0.012 0.028 0.013 0.132 0.048]
 [0.022 0.791 0.003 0.016 0.008 0.003 0.032 0.006 0.029 0.09 ]
 [0.079 0.011 0.322 0.1   0.161 0.073 0.161 0.05  0.025 0.018]
 [0.036 0.017 0.034 0.403 0.098 0.111 0.202 0.039 0.02  0.04 ]
 [0.033 0.01  0.034 0.064 0.587 0.031 0.152 0.058 0.017 0.014]
 [0.009 0.008 0.04  0.257 0.081 0.403 0.11  0.06  0.013 0.019]
 [0.002 0.01  0.012 0.043 0.045 0.012 0.841 0.013 0.006 0.016]
 [0.023 0.01  0.021 0.071 0.152 0.056 0.038 0.567 0.008 0.054]
 [0.08  0.049 0.007 0.024 0.011 0.005 0.018 0.007 0.762 0.037]
 [0.038 0.125 0.006 0.025 0.015 0.002 0.037 0.016 0.03  0.706]]
```
the following picture is displayed:
![confution_matrix](./examples/cifar10/conf_matrix.png)
