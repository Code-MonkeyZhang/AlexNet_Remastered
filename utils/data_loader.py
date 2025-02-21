# utils/data_loader.py

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageNetDataset(Dataset):
    """
    ImageNet dataset loader
    Directory structure:
    ILSVRC/
    ├── Data/CLS-LOC/
    │   ├── train/
    │   └── val/
    └── ImageSets/CLS-LOC/
        ├── train_cls.txt
        └── val.txt
    """
    def __init__(self, data_root, transform=None, train=True):
        """
        Args:
            data_root: ILSVRC/Data/CLS-LOC path
            transform: transforms to apply to images
            train: whether to load training or validation set
        """
        self.data_root = data_root
        self.transform = transform
        self.train = train
        
        # Get ILSVRC base directory
        self.base_dir = os.path.dirname(os.path.dirname(data_root))
        
        logger.info(f"Initializing ImageNet {'training' if train else 'validation'} dataset")
        logger.info(f"Data root: {data_root}")
        logger.info(f"Base directory: {self.base_dir}")
        
        # Load class mapping
        self._load_class_mapping()
        
        # Load images and labels
        self.images = []
        self.labels = []
        
        if self.train:
            self._load_training_set()
        else:
            self._load_validation_set()
            
        logger.info(f"Dataset loaded with {len(self.images)} images")

    def _load_class_mapping(self):
        """Load class names and create class to index mapping"""
        train_dir = os.path.join(self.data_root, 'train')
        # 获取所有 WordNet ID（目录名）
        self.classes = sorted([d for d in os.listdir(train_dir) 
                            if os.path.isdir(os.path.join(train_dir, d))])
        
        # 确保正好有1000个类别
        assert len(self.classes) == 1000, f"Found {len(self.classes)} classes, expected 1000"
        
        # 创建从 WordNet ID 到 0-999 索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _load_training_set(self):
        """Load training images and labels"""
        train_dir = os.path.join(self.data_root, 'train')
        
        logger.info("Loading training images...")
        for class_name in tqdm(self.classes, desc="Loading training classes"):
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                if self._is_valid_image(img_name):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def _load_validation_set(self):
        """Load validation images based on directory structure"""
        val_dir = os.path.join(self.data_root, 'val')
        logger.info(f"Loading validation images from: {val_dir}")
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")

        count = 0
        # 遍历验证集目录中的类别子目录
        for class_name in tqdm(sorted(os.listdir(val_dir)), desc="Loading validation classes"):
            class_path = os.path.join(val_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # 使用与训练集相同的类别索引
            if class_name not in self.class_to_idx:
                logger.warning(f"Warning: validation folder {class_name} not found in training classes")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # 加载该类别下的所有图片
            for img_name in os.listdir(class_path):
                if self._is_valid_image(img_name):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    count += 1
            
        logger.info(f"Found {count} validation images in {len(set(self.labels))} classes")

    def _is_valid_image(self, filename):
        """Check if a file is a valid image"""
        return filename.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            label = self.labels[idx]
            
            with Image.open(img_path) as img:
                image = img.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # 返回一个全0 tensor及一个特殊标签(-1)以便上层可以过滤
            dummy_tensor = torch.zeros(3, 224, 224)
            return dummy_tensor, -1


def create_data_loaders(data_root, train_transform, val_transform, 
                       batch_size=128, num_workers=4):
    """
    Create training and validation data loaders
    
    Args:
        data_root: path to ILSVRC/Data/CLS-LOC directory
        train_transform: transforms for training data
        val_transform: transforms for validation data
        batch_size: batch size for data loaders
        num_workers: number of worker processes for data loading
    """
    train_dataset = ImageNetDataset(
        data_root=data_root,
        transform=train_transform,
        train=True
    )
    
    val_dataset = ImageNetDataset(
        data_root=data_root,
        transform=val_transform,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader