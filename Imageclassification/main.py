import torch
from torchvision import datasets
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torchsummary
#from lenNet5 import Lenet5
from Net import ResNet18
from ImageDataset import ImageDataset

train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ])
test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def main():
    device = torch.device('cuda')
    batchsz = 128
    total_epoch = 100

    train_set = ImageDataset(transforms=train_transforms, is_val=False)
    train_loader = DataLoader(dataset=train_set, batch_size=batchsz, shuffle=True)

    test_set = ImageDataset(transforms=test_transforms, is_val=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batchsz, shuffle=True)


    model = ResNet18().to(device)
    '''model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    num_ftrs = model.fc.in_features  # 获取（fc）层的输入的特征数
    model.fc = nn.Linear(num_ftrs, 10)
    '''
    model.to(device)

    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(total_epoch):
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            label = torch.squeeze(label, dim=1)
            label = label.to(torch.long)
            #print('logits', logits)
            #print('label', label)
            #print('logits shape', logits.shape)
            #print('label shape', label.shape)
            # logits [b, 10]
            # label [b]
            loss = criteon(logits, label)
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #
        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                label = torch.squeeze(label)
                pred = logits.argmax(dim=1)
                #print('logits shape', logits.shape)
                #print('label shape', label.shape)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)

    # 打印各个类别准确率
    class_correct = list(0. for i in range(4))  # class_correct=[0.0, 0.0, 0.0, 0.0]
    class_total = list(0. for i in range(4))  # class_total=[0.0, 0.0, 0.0, 0.0]
    class_name = ['诺艾尔', '芙宁娜', '刻晴', '胡桃']
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels = torch.squeeze(labels)
            #print('label shape', labels.shape)
            outputs = model(images)  # outputs的维度是：4*10
            # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
            # 此时predicted的维度是：4*1
            _, predicted = torch.max(outputs, 1)
            # 此时c的维度：4将预测值与实际标签进行比较，且进行降维
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                #print('labels type', labels.type)
                label = labels[i].item()
                label = int(label)
                #print('label:', labels[i].item())
                #label = torch.int(label)

                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(4):
        print('Accuracy of %5s : %2d %%' % (class_name[i], 100 * class_correct[i] / class_total[i]))


if __name__=='__main__':
    main()

