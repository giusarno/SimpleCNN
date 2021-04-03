# SimpleCNN

# package to simplify using cnn networks



How to run Examples:

Run the MNIST example from console & windows:

# Install packages
change "C:/Users/GiuseppeSarno/IdeaProjects" with the root of your project.

set PYTHONPATH=%PYTHONPATH%;C:/Users/GiuseppeSarno/IdeaProjects/SimpleCNN/
cd C:\Users\GiuseppeSarno\IdeaProjects\SimpleCNN
pip install -r ../../requirements.txt

Known issues:
I hit this problem https://github.com/pytorch/pytorch/issues/23823 when installing "torch"
and fixed as on https://www.reddit.com/r/pytorch/comments/c6cllq/issue_installing_pytorch/ew27hih/?utm_source=share&utm_medium=web2x
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem = 1


# Run the model Generation

In this example we will be using the following parameters:
input(image) size   = 28x28
dimension           = 2 (we are dealing with 2 dimension input)
channels            =1  (only grayscale is used)
batch size          =10 (number of inputs will go through the model at the same time) 
filename            = themodel.py (output filename)
inputfile           = model.csv (input file with model definition)


python3 ./simplecnn/modelgen.py -i 28x28 -d 2 -c 1 -b 10 -f ./examples/mnist/themodel.py -v ./examples/mnist/model.csv


#Run the Training

python3 thetest.py

output:

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


the following picture is displaied:
![Loss Function](https://github.com/giusarno/SimpleCNN/blob/master/examples/mnist/loss.png)

