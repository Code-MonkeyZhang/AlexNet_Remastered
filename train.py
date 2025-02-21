import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.alexnet import AlexNet
from utils.data_loader import ImageNetDataset
from config import Config
import time
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset...")
    train_dataset = ImageNetDataset(
        Config.DATA_ROOT,
        transform=Config.TRAIN_TRANSFORM,
        train=True
    )
    
    print("Loading validation dataset...")
    val_dataset = ImageNetDataset(
        Config.DATA_ROOT,
        transform=Config.VAL_TRANSFORM,
        train=False
    )
    
    print("Creating data loaders...")
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
    
    print("Creating the model...")
    model = AlexNet(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print("Start training...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        
        # 使用tqdm包装训练数据加载器
        train_pbar = tqdm(train_loader, desc='Training', 
                         unit='batch', leave=True)
        
        for i, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 更新进度条描述
            if i % 10 == 9:  # 每10个batch更新一次显示的loss
                avg_loss = running_loss / 10
                train_pbar.set_description(f'Training (loss={avg_loss:.4f})')
                running_loss = 0.0
        
        print("\nStarting validation...")
        model.eval()
        correct = 0
        total = 0
        
        # 使用tqdm包装验证数据加载器
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', 
                          unit='batch', leave=True)
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新验证进度条描述
                current_acc = 100 * correct / total
                val_pbar.set_description(f'Validation (acc={current_acc:.2f}%)')
        
        val_accuracy = 100 * correct / total
        print(f'Validation Accuracy after epoch {epoch+1}: {val_accuracy:.2f}%')
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model_path = f'alexnet_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint: {model_path}")

if __name__ == '__main__':
    train()