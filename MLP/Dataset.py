import os
import glob
import random
import csv
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
import cv2



# 将所有图片的路径及其对应的标签存入csv文件，以便后面自定义数据类
'''
    root:生成csv文件的根目录
    lable_dic：创建一个空字典{key:value}，用来存放类别名和对应的标签

'''


def load_csv(root, lable_dic):
    images = []  # 存入所有图片的地址
    images_train, train_labels = [], []
    images_val, val_labels = [], []

    for category in lable_dic.keys():  # 遍历所有子目录，获得所有图片的路径
        # glob文件名匹配模式，不用遍历整个目录判断而获得文件夹下所有同类文件
        # 只考虑后缀为png,jpg,jpeg的图片
        images += glob.glob(os.path.join(root, category, '*.png'))
        images += glob.glob(os.path.join(root, category, '*.jpg'))
        images += glob.glob(os.path.join(root, category, '*.jpeg'))

    # print(len(images), images)  # 打印出图片的总数以及和所有图片路径名
    random.shuffle(images)  # 将所有图片顺序随机打乱(包括所用类别的，不只是在同类别中打乱)

    # 训练集80%  验证集20%
    images_train = images[:int(0.8 * len(images))]
    images_val = images[int(0.2 * len(images)):]

    train_filename = 'images_train.csv'
    if not os.path.exists(os.path.join(root, train_filename)):  # 如果不存在csv，则创建一个
        with open(os.path.join(root, train_filename), mode='w', newline='') as f:
            writer_train = csv.writer(f)
            for img in images_train:  # 遍历Datasets中存放的每一个图片的路径，如Datasets\\1\\00001.jpg
                category = img.split(os.sep)[-2]  # 用\\分隔，取倒数第二项作为类名 即子目录名称(获取当前图片的类别)
                label = lable_dic[category]  # 找到类名键对应的值，作为标签 (获取当前图片的类别所指定的标签) values
                writer_train.writerow([img, label])  # 写入csv文件，以逗号隔开，如：pokemon\\mewtwo\\00001.png, 2
            print('written into train_csv file:', train_filename)

    val_filename = 'images_val.csv'
    if not os.path.exists(os.path.join(root, val_filename)):  # 如果不存在csv，则创建一个
        with open(os.path.join(root, val_filename), mode='w', newline='') as f:
            writer_val = csv.writer(f)
            for img in images_val:  # 遍历Datasets中存放的每一个图片的路径，如Datasets\\1\\00001.jpg
                category = img.split(os.sep)[-2]  # 用\\分隔，取倒数第二项作为类名 即子目录名称(获取当前图片的类别)
                label = lable_dic[category]  # 找到类名键对应的值，作为标签 (获取当前图片的类别所指定的标签) values
                writer_val.writerow([img, label])  # 写入csv文件，以逗号隔开，如：pokemon\\mewtwo\\00001.png, 2
            print('written into train_val file:', val_filename)


'''
仅仅需要提供一个参数root，即放置数据集的根目录
'''


def load_pokemon(root):
    # 创建数字编码表
    label_dic = {}  # 创建一个空字典{key:value}，用来存放类别名和对应的标签
    for category in sorted(os.listdir(os.path.join(root))):  # 遍历根目录下不同类别，并排序
        if not os.path.isdir(os.path.join(root, category)):  # 如果不是文件夹，则跳过
            continue
        label_dic[category] = len(label_dic.keys())  # 给每个类别编码一个数字(为每个类别指定一个标签)
    load_csv(root, label_dic)  # 分割数据集并生成两个csv文件
    return root+'\images_train.csv',root+'\images_val.csv'



'''
重写Dataset类
'''


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, csv_file, transform=None, target_transform=None):  # 初始化一些需要传入的参数

        csv_reader = pd.read_csv(open(csv_file))
        imgs = csv_reader.values.tolist()  # 转换为列表

        self.imgs = imgs  # 列表的形式
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)  # 列表的长度

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]
        #img = Image.open(fn).convert('L')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)



        # if self.transform is not None:
        #     img = self.transform(img)  # 是否进行transform
        # return imgs
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容


