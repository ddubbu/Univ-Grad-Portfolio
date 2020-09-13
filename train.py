'''import libraries'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms  # 1 batch = (1, 784)
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
import numpy as np
#from torchsummary import summary

'''choose torch.device'''
print("is there cuda? :",  torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''data load'''
X_train, X_test, Y_train, Y_test = np.load("./data/30words_num=30000.npy", allow_pickle=True)

'''data rehspae and normalize'''
Y_train = np.array([Y_train])  # (1, 27000, 30)
Y_train = np.transpose(Y_train, (1, 0, 2))  # (27000, 1, 30)
Y_test = np.array([Y_test])  # (1, 27000, 30)
Y_test = np.transpose(Y_test, (1, 0, 2))  # (27000, 1, 30)

X_train = transforms.Normalize(mean=(0.0,), std=(1.0,))(torch.Tensor(X_train))
Y_train = transforms.Normalize(mean=(0.0,), std=(1.0,))(torch.Tensor(Y_train))

X_test = transforms.Normalize(mean=(0.0,), std=(1.0,))(torch.Tensor(X_test))
Y_test = transforms.Normalize(mean=(0.0,), std=(1.0,))(torch.Tensor(Y_test))

print("X_train.shape", X_train.shape)  # (27000, 173, 24)
print("Y_train.shape", Y_train.shape)  # (27000, 1, 30)
print("X_test.shape", X_test.shape)  # (3000, 173, 24)
print("Y_test.shape", Y_test.shape)  # (1, 27000, 30)


n_mfcc = 24
Tx = 173
num_class = 30

'''resnet 구현'''

'''3x3 convolution'''
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


'''Residual block'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

'''ResNet'''
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=num_class, p2=0.0):  # 10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)  # conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(4)  # 8
        self.fc = nn.Linear(64 * 11, num_classes)

        self.dropout2d = nn.Dropout2d(p=p2)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.dropout2d(out)
        out = self.layer2(out)
        out = self.dropout2d(out)
        out = self.layer3(out)
        # out = self.dropout2d(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return nn.LogSoftmax(dim=-1)(out)

'''Model Instance'''

''' hyper parameters '''
# total_batch_num = int(len(train_data) / batch_size)
epochs = 25
lr = 0.001
# momentum = 0.9
print_interval = 100
#drop_prob1 = 0.1 # ->fc에서는 덜 줄이고
drop_prob2 = 0.1  # ->conv에서
weight_decay = 1e-4

'''model, optimizer 초기화'''
model = ResNet(ResidualBlock, [2, 2, 2], p2=drop_prob2).to(device)  # [2, 2, 2] : blocks at each layer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

'''Model Summary'''
#summary(model,input_size=(1, Tx, n_mfcc))



''' Train Model '''
train_epoch_loss = []
train_epoch_acc = []
test_epoch_loss = []
test_epoch_acc = []

batch_size = 10

for epoch in range(epochs):

    ''' Train '''
    model.train()
    train_batch_loss = []
    train_batch_acc = []

    train_batch_num = int(len(X_train) / batch_size)
    # print("train_batch_num: ", train_batch_num)
    for batch_idx in range(train_batch_num):

        mini_batch_x = X_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]  # (50, 173, 24)
        mini_batch_x = mini_batch_x.unsqueeze(1)  # (50, 1, 173, 24)
        mini_batch_y = Y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        x, target = Variable(mini_batch_x).to(device), Variable(mini_batch_y.long()).to(device)

        optimizer.zero_grad()
        output = model(x)  # output.shape = (50, 30)
        target = target.squeeze(1)  # (50, 1, 30) -> (50, 30)
        target = torch.argmax(target, dim=1)  # -> (50)

        loss = F.nll_loss(output, target).to(device)

        loss.backward()  # calc gradients
        train_batch_loss.append(loss.item() / batch_size * 100)  # from tensor -> get value loss.item() or loss.data
        optimizer.step()  # update gradients
        prediction = output.argmax(dim=1, keepdims=True)
        accuracy = torch.true_divide(prediction.eq(target.view_as(prediction)).sum().data, batch_size) * 100
        train_batch_acc.append(accuracy)
        if batch_idx % print_interval == 0:
            print('epoch: {}\tbatch Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(
                epoch, batch_idx, train_batch_loss[batch_idx], train_batch_acc[batch_idx]))

    train_epoch_loss.append(np.sum(train_batch_loss) / train_batch_num)
    train_epoch_acc.append(np.sum(train_batch_acc) / train_batch_num)

    ''' Test '''
    model.eval()
    test_batch_loss = []
    test_batch_acc = []

    test_batch_num = int(len(X_test) / batch_size)
    with torch.no_grad():
        for batch_idx in range(test_batch_num):
            mini_batch_x = X_test[batch_idx * batch_size:(batch_idx + 1) * batch_size]  # (50, 173, 24)
            mini_batch_x = mini_batch_x.unsqueeze(1)  # (50, 1, 173, 24)
            mini_batch_y = Y_test[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            x, target = Variable(mini_batch_x).to(device), Variable(mini_batch_y.long()).to(device)

            optimizer.zero_grad()
            output = model(x)  # output.shape = (50, 30)
            target = target.squeeze(1)  # (50, 1, 30) -> (50, 30)
            target = torch.argmax(target, dim=1)  # -> (50)

            x, target = Variable(x).to(device), Variable(target).to(device)
            output = model(x)
            test_batch_loss.append(loss.item() / batch_size * 100)
            prediction = output.argmax(dim=1, keepdims=True)
            accuracy = torch.true_divide(prediction.eq(target.view_as(prediction)).sum().data, batch_size) * 100
            test_batch_acc.append(accuracy)

    test_epoch_loss.append(np.sum(test_batch_loss) / test_batch_num)
    test_epoch_acc.append(np.sum(test_batch_acc) / test_batch_num)

    # 중간 plot for checking overfitting
    x = np.arange(start=1, stop=len(train_epoch_loss) + 1, step=1)

    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, train_epoch_loss, label='train')
    plt.plot(x, test_epoch_loss, label='test')
    ax1.legend()
    ax1.set(ylabel="Loss", xlabel='epoch')

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, train_epoch_acc, label='train')
    plt.plot(x, test_epoch_acc, label='test')
    ax2.legend()
    ax2.set(ylabel="Accuracy", xlabel='epoch')

    plt.show()


''' save results to numpy '''
train_test_result = (train_epoch_loss, test_epoch_loss, train_epoch_acc, test_epoch_acc)
np.save("result.npy", train_test_result)

print("==================================")
print("train_epoch_loss:", train_epoch_loss)
print("test_epoch_loss:", test_epoch_loss)
print("train_epoch_acc", train_epoch_acc)
print("test_epoch_acc", test_epoch_acc)
print("==================================")
x = np.arange(start=1, stop=len(train_epoch_loss) + 1, step=1)

fig = plt.figure(figsize=(12, 3))
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(x, train_epoch_loss, label='train')
plt.plot(x, test_epoch_loss, label='test')
ax1.legend()
ax1.set(ylabel="Loss", xlabel='epoch')

ax2 = fig.add_subplot(1, 2, 2)
plt.plot(x, train_epoch_acc, label='train')
plt.plot(x, test_epoch_acc, label='test')
ax2.legend()
ax2.set(ylabel="Accuracy", xlabel='epoch')

plt.show()

# ''' inference를 위한 모델 저장  '''
model_save_type = ["all", "state_dict", "ckp"]  # 학습 재개를 위해서 ckp 저장 권장 // else state_dict
model_name = "./model/ResNet"
option = 2

if option == 0:
    model_path = model_name + '_all.pt'
    torch.save(model, model_path)

elif option == 1:
    model_path = model_name + '_state_dict.pt'
    torch.save(model.state_dict, model_path)

elif option == 2:
    model_path = model_name + '_ckp.tar'
    torch.save({
        'last_epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, model_path)
