import torch
from torch import nn, optim, argmax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


train_dataset = datasets.FashionMNIST(root="../dian", train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.FashionMNIST(root="../dian", train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size)
        for i in range(x.size(1)):
            y = torch.cat((x[:, i, :], h), dim=1)
            h = torch.tanh(self.layer1(y))
        out = self.layer2(h)
        return out


model = RNNModel(28, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):

    # 训练
    model.train()
    for data in train_loader:
        img, target = data
        img = img.reshape(-1, 28, 28)
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
            img = img.reshape(-1, 28, 28)
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
