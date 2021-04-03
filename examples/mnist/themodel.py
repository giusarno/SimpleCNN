import torch

class Model(torch.nn.Module):
    def __init__(self):
       super(Model, self).__init__()
       self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,5), stride=(1,1), padding=(0,0))
       self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2),padding=(0,0))
       self.relu_1 = torch.nn.ReLU()
       self.conv_2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5,5), stride=(1,1), padding=(0,0))
       self.dropout_1 = torch.nn.Dropout(p=0.5)
       self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2),padding=(0,0))
       self.relu_2 = torch.nn.ReLU()
       self.linear_1 = torch.nn.Linear(320,50)
       self.relu_3 = torch.nn.ReLU()
       self.dropout_2 = torch.nn.Dropout(p=0.5)
       self.linear_2 = torch.nn.Linear(50,10)
       self.logsoftmax_1 = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):
       x=self.conv_1(x)
       x=self.max_pool_1(x)
       x=self.relu_1(x)
       x=self.conv_2(x)
       x=self.dropout_1(x)
       x=self.max_pool_2(x)
       x=self.relu_2(x)
       x = x.view(-1,320)
       x=self.linear_1(x)
       x=self.relu_3(x)
       x=self.dropout_2(x)
       x=self.linear_2(x)
       x=self.logsoftmax_1(x)
       return x