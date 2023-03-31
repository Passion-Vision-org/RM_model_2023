import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn import Sequential
from PIL import Image
from torch.nn import functional as F

CNN = nn.Sequential(
      #input"28×28×1
    nn.Conv2d(1, 6, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),nn.ReLU(),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),nn.ReLU(),
    nn.Dropout2d(p=0.5),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
    nn.Dropout2d(p=0.1),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 5), nn.Softmax(dim=1)
)

MLP = nn.Sequential(
      #input"28×28×1
    nn.Flatten(),
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 50),
    nn.ReLU(),
    nn.Linear(50, 7),
    nn.Softmax(dim=1)
)

MLP_Simple = nn.Sequential(
      #input"28×28×1
    nn.Flatten(),
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 7), #relu后接softmax会不会出现都是0然后连接在sotfmax上，然后每个类别的分数都一样？
    nn.Softmax(dim=1)
)

class Net(nn.Module):
    def __init__(self):  # self是类Net的对象
        # 把类Net的对象self转换为类nn.Module的对象(转换为父类对象)，然后“被转换”的类nn.Module对象调用自己的init函数
        super(Net, self).__init__()  # 调用父类的.__init__()函数
        # 初始化层类型
        self.modle1 = Sequential(
            nn.Flatten(),
            nn.Linear(784, 200),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 定义前向传播
        x = self.modle1(x)
        return x #最后输出一个概率tensor


def train(model,train_loader,criterion,optimizer,epoch):
    model.train()  # 必备，将模型设置为训练模式
    writer = SummaryWriter("logs_MLP_Simple")
    for epoch in range(epoch):
        acc = 0
        total = 0
        # total_accuracy = 0
        total_test_loss = 0

        for step, data in enumerate(train_loader):  # 从数据加载器迭代一个batch的数据
            image, label = data
            total = len(label)  # 256+256+.....+160

            optimizer.zero_grad()  # 清除所有优化的梯度
            output = model(image)  # 喂入数据并前向传播获取输出

            #print(output)
            #print(label)
            # label先转换为独热码
            #one_hot = F.one_hot(label,num_classes = 5)
            loss = criterion(output, label)  # 调用损失函数计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 统计准确率
            predicted = torch.argmax(output.data, dim=1)
            acc = (predicted == label).sum()
            # total_accuracy = total_accuracy + acc



            # 每一轮绘制train_loss和train_acc以及test_acc
            if step % 10 == 0:  # 根据设置的显式间隔输出训练日志 每10个batchsize输出一次准确率
                Accuracy = float(acc) / float(total)
                print('Train Epoch: {} step：{} Loss: {:.6f} \t Accuracy:{:.6f} \t total：{} \t acc：{}'.format(
                    epoch, step, loss.item(), Accuracy, total, acc))  # 每一个step的准确率

                writer.add_scalar("train_Accuracy_MLP_Simple", Accuracy, epoch * 200 + step)
                writer.add_scalar("train_loss_Simple", loss.item(), epoch * 200 + step)

        # writer.add_scalar("train_Accuracy", float(total_accuracy) / step, epoch)
        # writer.add_scalar("train_loss", total_test_loss, epoch)

    writer.close()

#使用opencv读入并显示
def test(model,img_path):
    model.eval()
    #img_test = Image.open(img_path).convert('L')
    #plt.imshow(img_test)
    img_test = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    cv2.imshow('测试图片',img_test)
    #cv2.waitKey()
    img_tensor = torch.tensor(img_test) #转换为tensor
    img_tensor = img_tensor/255.0 #必须先进行标准化
    # print(img_tensor)
    img_tensor = img_tensor.reshape(1, 1, 28, 28)
    y_hat = model(img_tensor)
    print(y_hat)  # tensor([[0.1524, 0.1500, 0.1521, 0.3568, 0.1887]]
    result = torch.argmax(y_hat.data, dim=1)
    print(result.item() + 1)
#
#
def Change_model(model,output_name):
    print("----- 3 导出为onnx模型 -----")
    model.eval()  # 先将模型设置为推理模式
    # An example input you would normally provide to your model's forward() method
    # x为输入图像，格式为pytorch的NCHW格式；1为图像数一般不需要修改；3为通道数；224，224为图像高宽；
    #x = torch.rand(1,1,16, 36)
    x = torch.rand(1,1,28,28)
    # 模型输出名
    #output_name = r'D:\Datasets\000\Pulse.onnx'
    # Export the model
    # 导出为onnx模型
    # model为模型，x为模型输入，"mobilenet_v2.onnx"为onnx输出名，export_params表示是否保存模型参数
    # input_names为onnx模型输入节点名字，需要输入列表
    # output_names为onnx模型输出节点名字，需要输入列表；如果是多输出修改为output_names=["output1","output2"]
    torch_out = torch.onnx._export(model, x, output_name, verbose=True)
    print("模型导出成功")
