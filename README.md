# SimpleCNN

# package to simplify using cnn networks



How to run Examples:

Run the MNIST example from console & windows:

change "C:/Users/GiuseppeSarno/IdeaProjects" with the root of your project.

set PYTHONPATH=%PYTHONPATH%;C:/Users/GiuseppeSarno/IdeaProjects/SimpleCNN/
cd C:\Users\GiuseppeSarno\IdeaProjects\SimpleCNN
pip install -r ../../requirements.txt
python3 ./simplecnn/modelgen.py -i 28x28 -d 2 -c 1 -b 10 -f ./examples/mnist/themodel.py -v ./examples/mnist/model.csv




Known issues:
I hit this problem https://github.com/pytorch/pytorch/issues/23823 when installing "torch"
and fixed as on https://www.reddit.com/r/pytorch/comments/c6cllq/issue_installing_pytorch/ew27hih/?utm_source=share&utm_medium=web2x
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem = 1
