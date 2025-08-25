import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Data

from model import LeNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True
                              )

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0
                                       )

def test_model_process(model, test_dataloader):
    #   设定训练用到的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #  模型放到设备当中
    model = model.to(device)
    # 初始化参数准确度和损失函数
    test_loss = 0.0
    test_acc = 0.0
    # 只进行前向传播不进行运算
    with (torch.no.grad()):
        # 对测试数据进行验证
        for test_data_x ,test_data_y in test_dataloader:
            # 将数据放入设备中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            #将模型设置为验证模式
            model.eval()
            #将测试数据输入，进行前向传播，输出测试数据的预测结果
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标（概率最大的下标）
            pre_lab = torch.argmax(output, dim=1)
            #计算预测准确率的总之
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            #   当前用于样本的数量
            test_num += test_data_x.size(0)
    # 计算准确率的平均值
    test_acc = test_corrects.double().item() / test_num
    print('测试精确度为：' , test_acc)

    if '__name__' == '__main__':
        # 加载模型
        model = LeNet()
        # 将训练好的模型参数导入
        model.load_state_dict(torch.load('./model/LeNet.pth'))
        # 加载测试数据
        test_dataloader = test_data_process()
        # 加载模型测试的函数
        test_model_process(model, test_dataloader)
