from torchvision import transforms 

class Config:
    # 数据路径
    DATA_ROOT = '/media/yufeng/SSD/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'
    
    # 训练参数
    BATCH_SIZE = 128
    NUM_EPOCHS = 90
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # 数据预处理参数
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])