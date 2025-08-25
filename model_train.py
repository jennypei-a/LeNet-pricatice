import copy
from datetime import time
import time
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch
import torch.nn as nn
import pandas as pd


#   加载数据集
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True
                              )


    train_data,val_data = Data.random_split(train_data,
                                            [round(0.8*len(train_data)),
                                             round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=0
                                  )

    val_dataloader = Data.DataLoader(dataset=val_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=0
                                  )

    return train_dataloader,val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epoch):
    #   设定训练用到的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #   Adam是梯度下降法，使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #   损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    #  模型放到设备当中
    model = model.to(device)
    #   保存最好模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #   初始化参数
    #   最优精度
    best_acc = 0.0
    #   训练集损失列表
    train_loss_all = []
    #   验证集损失列表
    val_loss_all = []
    # 训练集精度列表
    train_acc_all = []
    # 验证集精度列表
    val_acc_all = []
    #   保存当前时间
    since = time.time()

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch,num_epoch-1))
        print("_"*10) # 打印10个"_"
        #   初始化训练集损失和准确度
        train_loss = 0.0
        train_corrects = 0.0
        #   初始化验证集损失和准确度
        val_loss = 0.0
        val_corrects = 0.0
        #   训练集和验证集的样本数量
        train_num = 0.0
        val_num = 0.0

        #   对一个mini-batch的数量计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            #   将训练数据放到设备中
            b_x = b_x.to(device)
            #   将训练标签放到设备中
            b_y = b_y.to(device)
            #   设置模型为训练模式
            model.train()
            #   前向传播，输入为一个batch，输出位一个batch中对应的预测
            output = model(b_x)
            #   查找每一行中最大值对应的行标（概率最大的下标）
            pre_lab = torch.argmax(output, dim=1)
            #   通过模型输出计算损失函数
            loss = criterion(output,b_y)
            #   将梯度初始化为0,防止梯度累计
            optimizer.zero_grad()
            #   反向传播计算
            loss.backward()
            #   根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            #   对损失函数累加
            train_loss += loss.item() * b_x.size(0)
            #   如果预测正确则准确度＋1
            train_corrects += torch.sum(pre_lab == b_y.data)
            #   当前用于样本的数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将验证数据放到设备中
            b_x = b_x.to(device)
            # 将验证标签放到设备中
            b_y = b_y.to(device)
            #   设置模型为验证模式、评估模式
            model.eval()
            # 前向传播，输入为一个batch，输出位一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标（概率最大的下标）
            pre_lab = torch.argmax(output, dim=1)
            # 通过模型输出计算损失函数
            loss = criterion(output, b_y)

            # 对损失函数累加
            val_loss += loss.item() * b_x.size(0)
            # 验证样本如果预测正确则准确度＋1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本的数量
            val_num += b_x.size(0)

        # 计算训练集每一次迭代loss的准确率
        train_loss_all.append(train_loss / train_num)
        #   计算训练集的平均精度
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 计算验证集每一次迭代loss的准确率
        val_loss_all.append(val_loss / val_num)
        #   计算验证集的训练精度平均值
        val_acc_all.append(val_corrects.double().item() / val_num)
        #   每次打印该轮次的最后一个数组值
        print('{} train loss: {:.4f} train Acc: {:.4f}'.format(epoch, train_loss_all[-1],train_acc_all[-1]))
        print('{} val loss: {:.4f} val Acc: {:.4f}'.format(epoch, val_loss_all[-1],val_acc_all[-1]))

        #   选取当前的准确度，判断是否大于最高准确度
        if val_acc_all[-1] > best_acc:
            #   将该准确度赋值给最高准确度
            best_acc = val_acc_all[-1]
            #   保存当前的参数
            best_model_wts = copy.deepcopy(model.state_dict())
        #   训练耗时
        time_use = time.time() - since
        print("训练和验证集的耗费时间 {:.0f} m {:.0f} s".format(time_use//60, time_use%60))

    #   加载模型最高参数,pth权重文件的后缀
    torch.save(best_model_wts, 'F:/PycharmProjects/LeNet/best_model.pth')

    train_process = pd.DataFrame(data={'epoch':  list(range(num_epoch)),
                                       'train_loss_all': train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_acc_all})

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-',label = 'train loss')
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'go-', label='train acc')
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'ys-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.show()



if __name__ == '__main__':
    # 将模型实例化
    LeNet = LeNet()
    # 训练数据
    train_dataloader, val_dataloader = train_val_data_process()
    # 训练过程
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 10)
    # 画图
    matplot_acc_loss(train_process)