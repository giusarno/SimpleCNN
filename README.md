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
> python3 thetest.py
```
Output:
```
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.395239
Train Epoch: 1 [640/60000 (1%)] Loss: 2.281763
Train Epoch: 1 [1280/60000 (2%)]        Loss: 2.231858
Train Epoch: 1 [1920/60000 (3%)]        Loss: 2.165063
Train Epoch: 1 [2560/60000 (4%)]        Loss: 2.135278
Train Epoch: 1 [3200/60000 (5%)]        Loss: 2.046879
Train Epoch: 1 [3840/60000 (6%)]        Loss: 1.832578
..............................................
..............................................
Train Epoch: 5 [55680/60000 (93%)]      Loss: 0.189094
Train Epoch: 5 [56320/60000 (94%)]      Loss: 0.117050
Train Epoch: 5 [56960/60000 (95%)]      Loss: 0.081903
Train Epoch: 5 [57600/60000 (96%)]      Loss: 0.066808
Train Epoch: 5 [58240/60000 (97%)]      Loss: 0.108263
Train Epoch: 5 [58880/60000 (98%)]      Loss: 0.081022
Train Epoch: 5 [59520/60000 (99%)]      Loss: 0.092410
```
the following picture is displaied:


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
the following picture is displaied:

![confusion matrix](https://github.com/giusarno/SimpleCNN/blob/master/examples/mnist/conf_matrix.png)
