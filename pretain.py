# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torchvision.models import resnet18,wide_resnet50_2
# import os

# # 定义训练函数
# def train_model(model, train_loader, test_loader, device, num_epochs=50, save_path='./prototypes/pretrained_models', model_name='resnet18_cifar.pth'):
#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # 创建保存路径
#     os.makedirs(save_path, exist_ok=True)
#     model_path = os.path.join(save_path, model_name)

#     # 训练模型
#     model = model.to(device)
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             # 前向传播
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         # 打印每个 epoch 的损失
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

#         # 测试模型
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         accuracy = 100 * correct / total
#         print(f"Test Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")

#     # 保存模型权重
#     torch.save(model.state_dict(), model_path)
#     print(f"Model saved to {model_path}")

# # 定义主函数
# def main():
#     # 配置设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 的均值和标准差
#     ])


#     # 加载 CIFAR-10 数据集
#     train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
#     test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

#     # 选择模型（ResNet-18 或 WideResNet）
#     model_choice = 'resnet18'  # 可选 'resnet18' 或 'wideresnet'

#     if model_choice == 'resnet18':
#         # 使用 ResNet-18
#         model = resnet18(pretrained=False)  # 不加载 ImageNet 权重
#         model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 有 10 个类别
#         model_name = 'resnet18_cifar.pth'
#     elif model_choice == 'wideresnet':
#         # 使用 WideResNet
#         model = WideResNet(depth=28, widen_factor=10, num_classes=10)
#         model_name = 'wideresnet_cifar.pth'
#     else:
#         raise ValueError("Invalid model choice. Use 'resnet18' or 'wideresnet'.")

#     # 训练模型
#     train_model(model, train_loader, test_loader, device, num_epochs=50, save_path='./prototypes/pretrained_models', model_name=model_name)

# if __name__ == '__main__':
#     main()



import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
 
class ResBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride=2):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1) # ! (h-3+2)/2 + 1 = h/2 图像尺寸减半
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1) # ! h-3+2*1+1=h 图像尺寸没变化
        self.bn2 = nn.BatchNorm2d(ch_out)
 
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride), # ! 这句话是针对原图像尺寸写的，要进行element wise add 
                                                            # ! 因此图像尺寸也必须减半，(h-1)/2+1=h/2 图像尺寸减半
            nn.BatchNorm2d(ch_out)
        )
 
    
    def forward(self,x):
        out = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # short cut
        # ! element wise add [b,ch_in,h,w] [b,ch_out,h,w] 必须当ch_in = ch_out时才能进行相加
        out = x + self.extra(out) # todo self.extra强制把输出通道变成一致
        return out
 
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1), # ! 图像尺寸不变
            nn.BatchNorm2d(64)
        )
        # 4个ResBlock
        #  [b,64,h,w] --> [b,128,h,w]
        self.block1 = ResBlock(64,128)
        #  [b,128,h,w] --> [b,256,h,w]
        self.block2 = ResBlock(128,256)
        #  [b,256,h,w] --> [b,512,h,w]
        self.block3 = ResBlock(256,512)
        #  [b,512,h,w] --> [b,512,h,w]
        self.block4 = ResBlock(512,512)
 
        self.outlayer = nn.Linear(512,10)
 
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        # [b,64,h,w] --> [b,1024,h,w]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # print("after conv:",x.shape)
        #[b,512,h,w] --> [b,512,1,1]
        x = F.adaptive_avg_pool2d(x,[1,1])
        #flatten
        x = x.view(x.shape[0],-1)
        x = self.outlayer(x)
        return x
    

 
def get_acc(output, label):
    total = output.shape[0]
    pred_label = output.argmax(dim=1)
    num_correct = (pred_label == label).float().sum().item()
    return num_correct / total
 
def main():
    batchsz=64
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 的均值和标准差
    ])


    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)

    cifar10_train= DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
    cifar10_test = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)

 
    device = torch.device('cuda:0')
    model = ResNet18()
    model.to(device)
    # print(model)
    criteon = nn.CrossEntropyLoss().to(device) #包含了softmax操作
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(10):
        train_loss = 0
        train_acc = 0
        model.train()
        for batchidx,(x,label) in enumerate(cifar10_train):
            #[b,3,32,32]
            #[b]
            x,label = x.to(device),label.to(device)
            # y_:[b,10]
            # label:[b]
            y_ = model(x)
            loss = criteon(y_,label)
 
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
            train_acc += get_acc(y_,label)
 
        model.eval()
        print("epoch:%d,train_loss:%f,train_acc:%f"%(epoch, train_loss / len(cifar10_train),
            train_acc / len(cifar10_train)))  
        torch.save(model,'./prototypes/pretrained_models/ResNet18_%d.pth'%(epoch))
 
 
if __name__ == "__main__":
    main()