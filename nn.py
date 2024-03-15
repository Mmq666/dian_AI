import torch
from torch import optim, argmax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn


# 规定数字0为正例，1~9数字为负例
# 准确率
def accuracy(tp, tn, fp, fn):
    ans = (tp + tn) / (tp + tn + fp + fn)
    return ans


# 精确率
def precision(tp, fp):
    ans = tp / (tp + fp)
    return ans


# 召回率
def recall(tp, fn):
    ans = tp / (tp + fn)
    return ans


# F1值
def f1(precision, recall):
    ans = 2 * precision * recall / (precision + recall)
    return ans


transform = transforms.Compose({
    transforms.ToTensor(),  # 转为Tensor，范围改为0-1
    transforms.Normalize((0.1307,), (0.3081))  # 数据归一化，即均值为0，标准差为1
})

train_dataset = datasets.MNIST(root="../dian", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root="../dian", train=False, transform=transform, download=False)

# print(train_dataset)
# print(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(784, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dense(x)
        return x


model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

for epoch in range(10):

    # 训练
    model.train()
    for data in train_loader:
        img, target = data
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        correct = 0
        for data in test_loader:
            img, target = data
            output = model(img)
            predict = argmax(output, dim=-1)
            TP += ((predict == 0) & (target == 0)).sum().item()
            TN += ((predict == target) & (target != 0)).sum().item()
            FP += ((predict == 0) & (target != 0)).sum().item()
            FN += ((predict != 0) & (target == 0)).sum().item()

        acc = accuracy(TP, TN, FP, FN)
        pre = precision(TP, FP)
        rec = recall(TP, FN)
        F1 = f1(pre, rec)

        print("epoch: {} --> accuracy: {:.6f} - precision: {:.6f} - recall: {:.6f} - f1: {:.6f}".format(
            epoch + 1, acc, pre, rec, F1))
