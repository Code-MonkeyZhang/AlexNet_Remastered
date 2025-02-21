import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.alexnet import AlexNet
from utils.data_loader import ImageNetDataset
from config import Config
import time

def train():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集和数据加载器
    train_dataset = ImageNetDataset(
        Config.DATA_ROOT,
        transform=Config.TRAIN_TRANSFORM,
        train=True
    )
    
    val_dataset = ImageNetDataset(
        Config.DATA_ROOT,
        transform=Config.VAL_TRANSFORM,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = AlexNet(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练循环
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 打印训练状态
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Time: {time.time()-start_time:.2f}s')
                running_loss = 0.0
                start_time = time.time()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'alexnet_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()