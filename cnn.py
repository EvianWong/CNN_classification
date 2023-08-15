import numpy as np
import pandas as pd
import os

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torch.optim as optim


# 准备数据
def load_data():
    data_1 = pd.read_csv('../../data/MovieGenre.csv', encoding='gbk')
    data_1.dropna(inplace=True)
    img_title = data_1['Title']
    genre = data_1['Genre']
    score = data_1['IMDB Score']

    data_2 = pd.read_csv('../../data/rgb_index2.csv', encoding='utf-8')
    index = data_2['post_index']
    return index, genre, score, img_title


def get_data_y():
    index, genre, _, _ = load_data()

    genre_1 = genre.str.split('|')
    main_genre = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Romance', 'Thriller', 'Unknown']

    genre_list = np.unique(main_genre)
    data_y = []
    for i in index:
        array = np.zeros(len(genre_list), dtype=float)
        if genre_1[i][0] in genre_list:
            array_1 = np.int32(genre_1[i][0] == genre_list)
        else:
            array_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        array += array_1
        data_y.append(array)
    
    return data_y, genre_list


# pytorch模型
# 读入自己的dataset
class CustomImageDataset(Dataset):
    def __init__(self, mode, target_data, annotations_file, img_dir, transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.target_data = target_data
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        if mode == 'train':  # 80%
            self.img_labels = self.img_labels[:int(0.8 * len(self.img_labels))]
            self.target_data = self.target_data[:int(0.8 * len(self.target_data))]
        else:
            self.img_labels = self.img_labels[int(0.8 * len(self.img_labels)):]
            self.target_data = self.target_data[int(0.8 * len(self.target_data)):]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 1])+'.jpg')
        image_1 = read_image(img_path)
        image = image_1.float()
        label = torch.tensor(self.target_data[idx], dtype=torch.float)
        image = self.transform(image)
        return image, label


# cnn模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 15, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(15, 30, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(30, 30, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(3)
        self.fc = torch.nn.Linear(1470, 9)

    def forward(self, x):
        batch_size = x.size(0)
        x = func.leaky_relu(self.pooling(self.conv1(x)))           # 3*215*215 -> 15*213*213 -> 15*71*71
        x = func.leaky_relu(self.pooling(self.conv2(x)))           # 30*70*70 -> 15*69*69 -> 30*23*23
        x = func.leaky_relu(self.pooling(self.conv3(x)))           # 30*23*23 -> 30*21*21 -> 30*7*7
        x = x.view(batch_size, -1)                                 # 30*7*7 = 1470
        x = self.fc(x)                                             # 全连接层 1470 -> 9
        return x

#
# def weight_init(m):  # 初始化权重
#     if isinstance(m, torch.nn.Conv2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#     elif isinstance(m, torch.nn.Linear):
#         m.weight.data.normal_(0, 0.02)
#         m.bias.data.zero_()


# 训练
def train(epoch_in):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_dataloader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 50 == 49:
            print('[%d, %5d] loss:%.3f' % (epoch_in+1, batch_idx+1, running_loss/50))
            running_loss = 0.0


# 误差计算
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            _, target_label = torch.max(labels.data, dim=1)
            total += labels.size(0)
            correct += (predicted == target_label).sum().item()
    print('Accuracy on test set: %d %%' % (100*correct/total))


if __name__ == '__main__':
    y_data, all_genre = get_data_y()
    batchsize = 1
    # 图片normalize
    pic_transform = transforms.Compose([
        transforms.Resize([215, 215]),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0.0,), std=(256,))  # 均值，标准差
    ])
    my_train_dataset = CustomImageDataset(mode='train', target_data=y_data,
                                          annotations_file='../../data/rgb_index2.csv',
                                          img_dir='../../data/poster/', transform=pic_transform)
    my_test_dataset = CustomImageDataset(mode='test', target_data=y_data,
                                         annotations_file='../../data/rgb_index2.csv',
                                         img_dir='../../data/poster/', transform=pic_transform)
    train_dataloader = DataLoader(my_train_dataset, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(my_test_dataset, batch_size=batchsize, shuffle=True)

    model = Net()
    # model.apply(weight_init)  # 加载权重
    # 损失优化
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        train(epoch)
        test()
