import torch

class Model(torch.nn.Module):
    def __init__(self):
       super(Model, self).__init__()
       self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
       self.relu_1 = torch.nn.ReLU()
       self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2),padding=(0,0))
       self.conv_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
       self.relu_2 = torch.nn.ReLU()
       self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2),padding=(0,0))
       self.linear_1 = torch.nn.Linear(400,120)
       self.relu_3 = torch.nn.ReLU()
       self.linear_2 = torch.nn.Linear(120,84)
       self.relu_4 = torch.nn.ReLU()
       self.linear_3 = torch.nn.Linear(84,10)
       self.logsoftmax_1 = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):
       x=self.conv_1(x)
       x=self.relu_1(x)
       x=self.max_pool_1(x)
       x=self.conv_2(x)
       x=self.relu_2(x)
       x=self.max_pool_2(x)
       x = x.view(-1,400)
       x=self.linear_1(x)
       x=self.relu_3(x)
       x=self.linear_2(x)
       x=self.relu_4(x)
       x=self.linear_3(x)
       x=self.logsoftmax_1(x)
       return x