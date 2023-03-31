import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms  #用到了的库就会高亮
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim #优化器
from sklearn.model_selection import KFold   #引入K折叠交叉
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import onnx

#引用其他文件的函数
from Dataset import *
from Moudle import *
from SVM import *
from HOG import *
from SVM_test import *


# #***************************1.加载数据集****************************/
# root_path = r'D:\Datasets\000_Binary' #D:\Datasets\000_Binary   C:\\Users\\zsy\\Desktop\\base\\000_Binary
# train_csv, val_csv = load_pokemon(root_path)
#
#
# #/**************************2.数据预处理***************************/
# train_transformer = transforms.Compose([
#     transforms.ToTensor(),
#     #transforms.Normalize([0.2954], [0.4562])
# ])
#
# train_data=MyDataset(train_csv,transform=train_transformer)
# val_data=MyDataset(val_csv,transform=train_transformer)
#
# dataset = ConcatDataset([train_data, val_data]) #合并训练集和验证集
#
# Train_data_loader = DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0)  #均值为tensor([0.0376]) 标准差为 tensor([0.0316])  这里拼接起来了训练集和验证集
# Val_data_loader = DataLoader(val_data,batch_size=128,shuffle=True,num_workers=0) #均值为tensor([0.0378]) 标准差为 tensor([0.0317])


#/*****************************3.加载模型**********************/
# #这里是MLP
# model_MLP = torch.load('MLP_Simple16.pth')
# criterion= nn.CrossEntropyLoss()  #这里使用的是交叉熵损失
# optimizer= optim.SGD(model_MLP.parameters(),lr=0.01,momentum=0.9)  #随机梯度下降
# epoch = 10
# train(model_MLP,Train_data_loader,criterion,optimizer,epoch)
# torch.save(model_MLP,'MLP_Simple17.pth')     #保存模型

# print(model_MLP)



# # 这里是HOG+SVM
# model = None
# model = joblib.load('D:\\DA\\SVM_HOG1.pkl')
# SVM_main(model, train_data, val_data, 'SVM_HOG2')

# 测试HOG+SVM
# clf = joblib.load('D:\\DA\\SVM_HOG1.pkl')  # 加载模型
# img_file = 'D:\\Datasets\\000_Binary\\1\\88.png'   #测试图片位置
# hog = HogDescriptor(pixels_per_cell=4, bin_size=9)
# fn = hog.get_feat_test(image_root=img_file)
# print(len(fn))
# result = clf.decision_function(fn)
# print(result)
# print(np.argmax(result) + 1)



#/*****************************4.测试模型**********************/
# writer = SummaryWriter("logs_train")
# model_MLP = torch.load('MLP_Simple17.pth')
# val_dataloader = Train_data_loader
# criterion= nn.CrossEntropyLoss()
# total_test_loss = 0
# predicted = 0
# acc = 0
# total_accuracy = 0
# with torch.no_grad():
#     for step, data in enumerate(val_dataloader):
#         imgs, label = data
#         total = len(label)
#         outputs = model_MLP(imgs)
#         loss = criterion(outputs, label)
#         total_test_loss = total_test_loss + loss.item() #测试集上的Loss
#
#         # 统计准确率
#         predicted = torch.argmax(outputs.data, dim=1)
#         acc = (predicted == label).sum()
#
#         if step % 10 == 0:
#             writer.add_scalar("test_loss", total_test_loss, step)
#             Accuracy = float(acc) / float(total)
#             writer.add_scalar("test_accuracy", Accuracy, step)
#             total_test_loss = 0
#
# writer.close()

# 单图片传入
model_MLP = torch.load('MLP_Simple17.pth')
test_path = r'C:\Users\zsy\Desktop\base\cut\a3.png' # D:\Datasets\000_Binary\1\76.png 路径里面不能有中文，否则opencv读取不出来 C:\Users\zsy\Desktop\base\cut
test(model_MLP,test_path)
# print("这里是测试") #换一种思路，只要负类能将真类的置信度限制在0.9以下就行，不一定负类要达到0.9以上
# Change_model(model_MLP,r'D:\DA\MLP_Simple1.onnx') #D:\Datasets\000_Binary\MLP_Simple.onnx

# #打乱后的数据集
# writer = SummaryWriter("logs")
# for step, data in enumerate(Train_data_loader):  # 从数据加载器迭代一个batch的数据
#     image, label = data
#     writer.add_images("show", image, step)
# writer.close()

#/*****************************5.onnx使用**********************/
# model = onnx.load("MLP_Simple.onnx")
# model = cv2.dnn.readNetFromONNX("MLP_Simple1.onnx")
# # print(model)
#
# test_path = r'C:\Users\zsy\Desktop\base\cut\a0.png'
#
# # model.eval()
# img_test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
# # cv2.imshow('测试图片',img_test)
# print(model)
# blob = cv2.dnn.blobFromImage(img_test)  # 由图片加载数据 这里还可以进行缩放、归一化等预处理
# model.setInput(blob)  # 设置模型输入
# out = model.forward()
#
# print(out)
# print(np.argmax(out) + 1)

