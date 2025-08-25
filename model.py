import torch
from torch import nn
# nn提供神经网络模块，卷积层、池化层等内容
from torchsummary import summary

class LeNet(nn.Module):
    # 定义方法，初始化神经网络的各个层
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        self.sig = nn.Sigmoid()
        self.s1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 初始化平坦层，将多维张量展开为1维
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(400, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 10)
    # 定义方法输入x后实现前向传播（告诉pytorch如何实现这个神经网络），在module中定义了用forward来实现数据流
    def forward(self, x):
        x = self.c1(x)
        x = self.sig(x)
        x = self.s1(x)
        x = self.c2(x)
        x = self.s2(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return x
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
        #判断是否支持cpu还是cpu
    model = LeNet().to(device)
    #print(summary(model, (1, 28, 28)))