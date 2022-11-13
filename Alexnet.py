import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json 
import zipfile
import os 

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print(device)

class AlexNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 11, padding = 2 ,stride = 4)
        self.conv2 = nn.Conv2d(in_channels = 64 , out_channels = 192, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 192 , out_channels = 384, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 384 , out_channels = 256, kernel_size = 5, padding = 2)
        self.conv5 = nn.Conv2d(in_channels = 256 , out_channels = 256, kernel_size = 5, padding = 2)

        self.fc1 = nn.Linear(in_features = 256 * 6 * 6, out_features = 4096)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 1024)
        self.fc3 = nn.Linear(in_features = 1024, out_features = num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size = 3,stride = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 3,stride = 2)
        x = F.relu(self.conv3)
        x = F.relu(self.conv4)
        x = F.relu(self.conv5)
        x = F.max_pool2d(x, kernel_size = 3,stride = 2)
        x = torch.flatten(x,start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.5)
        x = F.relu(self.fc2(x)) 
        x = F.dropout(x, p = 0.5)
        x = F.relu(self.fc3(x))
        return x

# Parameters
#===================================================
batch_size = 128
num_epochs = 30
lr = 0.001

num_classes=2
model = AlexNet(num_classes)

if CUDA:
    model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# transform
#===================================================
transform = transforms.Compose(
    [transforms.Resize(size =(227, 227)),
     transforms.CenterCrop(224),
     transforms.RandomRotation(20),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5),(0.5)),]
)
# Datasets
#====================================================
train_dataset = datasets.ImageFolder(root = 'kaggle/dsets/datasets/training_set',transform = transform)
test_dataset = datasets.ImageFolder(root = "kaggle/dsets/dataset/test_set",transform = transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
# Train
# ====================================================
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_train = 0 
    correct_train = 0 
    train_loss = 0 
    
    for batch_idx,(data, target) in enumerate(train_loader):
        data,target = Variable(data),Variable(target)
        
        if CUDA:
            data, target = data.cuda(),target.cuda()
            
        optimizer.zero_grad() # 梯度規0
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        predicted = torch.max(output.data, 1)[1]
        total_train += len(target)
        correct_train += sum((predicted == target).float())
        train_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print("Train Epoch:{}/{}[iter: {}{}], acc : {:.6f}, loss : {:.6f} ".format(
            epoch + 1 ,num_epochs, batch_idx + 1,len(train_loader),
            correct_train / float((batch_idx + 1) * batch_size),
            train_loss / float((batch_idx + 1 )* batch_size)))
            
    train_acc_ = 100 * (correct_train / float(total_train))
    train_loss_ = train_loss / total_train 
    
    return train_acc_, train_loss_
# Test 
# ===========================================================
def validate(test_loader, model, criterion, epoch):
    model.eval()
    total_test = 0 
    correct_test = 0 
    test_loss = 0 
    
    for batch_idx,(data, target) in enumerate(test_loader):
        data,target = Variable(data),Variable(target)
        
        if CUDA:
            data, target = data.cuda(),target.cuda()
            
        # Test 沒有 優化器   
        #optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        #loss.backward()
        #optimizer.step()
        
        predicted = torch.max(output.data, 1)[1]
        total_test += len(target)
        correct_test += sum((predicted == target).float())
        test_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print("Train Epoch:{}/{}[iter: {}{}], acc : {:.6f}, loss : {:.6f} ".format(
            epoch + 1 ,num_epochs, batch_idx + 1,len(test_loader),
            correct_test / float((batch_idx + 1) * batch_size),
            test_loss / float((batch_idx + 1 )* batch_size)))
            
    test_acc_ = 100 * (correct_test / float(total_test))
    test_loss_ = test_loss / total_test 
    
    return test_acc_, test_loss_
# Training loop 
# =============================================================
def training_loop(model, criterion, optimizer, train_loader, test_loader):
    # set objects for storing metrics
    total_train_loss = []
    total_test_loss = []
    total_train_accuracy = []
    total_test_accuracy = []
 
    # Train model
    for epoch in range(num_epochs):
        # training
        train_acc_, train_loss_ = train(train_loader, model, criterion, optimizer, epoch)
        total_train_loss.append(train_loss_)
        total_train_accuracy.append(train_acc_)

        # validation
        with torch.no_grad():
            test_acc_, test_loss_ = validate(test_loader, model, criterion, epoch)
            total_test_loss.append(test_loss_)
            total_test_accuracy.append(test_acc_)

        print('==========================================================================')
        print("Epoch: {}/{}， Train acc： {:.6f}， Train loss： {:.6f}， Test acc： {:.6f}， Test loss： {:.6f}".format(
               epoch+1, num_epochs, 
               train_acc_, train_loss_,
               test_acc_, test_loss_))
        print('==========================================================================')

    print("====== END ==========")

    return total_train_loss, total_test_loss, total_train_accuracy, total_test_accuracy