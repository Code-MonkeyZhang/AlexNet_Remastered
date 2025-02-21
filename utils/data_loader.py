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
        self.classes = sorted([d for d in os.listdir(train_dir) 
                             if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        logger.info(f"Found {len(self.classes)} classes")

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
        """Load validation images and labels"""
        val_dir = os.path.join(self.data_root, 'val')
        val_anno_file = os.path.join(self.base_dir, 'ImageSets/CLS-LOC/val.txt')
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        if not os.path.exists(val_anno_file):
            raise FileNotFoundError(f"Validation annotation file not found: {val_anno_file}")
            
        # Read validation annotations
        logger.info("Loading validation annotations...")
        val_labels = {}
        with open(val_anno_file, 'r') as f:
            for line in f:
                # Format: ILSVRC2012_val_00000001 1
                img_name, label = line.strip().split()
                img_name = img_name + '.JPEG'
                # Convert 1-based label to 0-based
                label = int(label) - 1
                val_labels[img_name] = label
        
        # Load validation images
        logger.info("Loading validation images...")
        for img_name in tqdm(sorted(os.listdir(val_dir)), desc="Loading validation images"):
            if self._is_valid_image(img_name):
                if img_name in val_labels:
                    label = val_labels[img_name]
                    if 0 <= label < len(self.classes):
                        img_path = os.path.join(val_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(label)
                    else:
                        logger.warning(f"Invalid label {label} for image {img_name}")
                else:
                    logger.warning(f"No annotation found for image {img_name}")

    def _is_valid_image(self, filename):
        """Check if a file is a valid image"""
        return filename.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
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
            # Return a random valid image instead
            return self[torch.randint(len(self), (1,)).item()]

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