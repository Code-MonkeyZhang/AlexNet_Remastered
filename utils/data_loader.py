# utils/data_loader.py

import os
import pickle
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImageNetDataset(Dataset):
    def __init__(self, data_root, transform=None, train=True, cache_file=None):
        """
        Args:
            data_root: ILSVRC/Data/CLS-LOC 路径
            transform: 对图像进行的预处理
            train: 是否加载训练集
            cache_file: 缓存文件路径，如果为 None 则默认存放在 data_root 下
        """
        self.data_root = data_root
        self.transform = transform
        self.train = train
        
        # 获取 ILSVRC 的基础目录
        self.base_dir = os.path.dirname(os.path.dirname(data_root))
        
        logger.info(f"Initializing ImageNet {'training' if train else 'validation'} dataset")
        logger.info(f"Data root: {data_root}")
        logger.info(f"Base directory: {self.base_dir}")
        
        # 加载类别映射
        self._load_class_mapping()
        
        # 如果没有提供缓存文件路径，则默认生成一个
        if cache_file is None:
            cache_file = os.path.join(self.data_root, f'{"train" if train else "val"}_cache.pkl')
        self.cache_file = cache_file
        
        # 尝试加载缓存
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached dataset from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.images, self.labels = pickle.load(f)
        else:
            # 如果没有缓存则遍历目录加载数据
            self.images = []
            self.labels = []
            
            if self.train:
                self._load_training_set()
            else:
                self._load_validation_set()
            
            logger.info(f"Dataset loaded with {len(self.images)} images")
            # 保存缓存
            with open(self.cache_file, 'wb') as f:
                pickle.dump((self.images, self.labels), f)

    def _load_class_mapping(self):
        train_dir = os.path.join(self.data_root, 'train')
        # 获取所有 WordNet ID（目录名）
        self.classes = sorted([d for d in os.listdir(train_dir) 
                               if os.path.isdir(os.path.join(train_dir, d))])
        assert len(self.classes) == 1000, f"Found {len(self.classes)} classes, expected 1000"
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _load_training_set(self):
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
        val_dir = os.path.join(self.data_root, 'val')
        logger.info(f"Loading validation images from: {val_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        for class_name in tqdm(sorted(os.listdir(val_dir)), desc="Loading validation classes"):
            class_path = os.path.join(val_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in self.class_to_idx:
                logger.warning(f"Warning: validation folder {class_name} not found in training classes")
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                if self._is_valid_image(img_name):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def _is_valid_image(self, filename):
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