import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

import config
# from preprocess import preprocess_image
from preprocess import preprocess_image_onlyresize  # 使用新的预处理函数

class GameDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        label_counts = {label: 0 for label in range(len(image_dirs))}
        
        red_flash_count = 0
        red_flash_label = 1  # RED_FLASH 标签的索引
        for img_name in os.listdir(image_dirs[red_flash_label]):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(image_dirs[red_flash_label], img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = preprocess_image_onlyresize(image, target_size=(config.INPUT_SIZE, config.INPUT_SIZE_HIGH))
                image = np.expand_dims(image, axis=0)  # 添加通道维度，使其变为 (1, H, W)
                image = np.transpose(image, (1, 2, 0))  # 转换为 (H, W, 1)
                if self.transform:
                    image = self.transform(image)
                else:
                    image = torch.tensor(image, dtype=torch.float32)
                self.images.append(image)
                self.labels.append(red_flash_label)
                label_counts[red_flash_label] += 1
                red_flash_count += 1

        for label, image_dir in enumerate(image_dirs):
            if label == red_flash_label:
                continue  # RED_FLASH 标签的数据已经加载
            
            for img_name in os.listdir(image_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    if label == 2 and label_counts[label] >= red_flash_count * 2:
                        break  # NO_FLASH 标签的图像数量限制为 RED_FLASH 标签图像数量的2倍
                    
                    img_path = os.path.join(image_dir, img_name)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    image = preprocess_image(image, target_size=(config.INPUT_SIZE, config.INPUT_SIZE_HIGH))
                    image = np.expand_dims(image, axis=0)  # 添加通道维度，使其变为 (1, H, W)
                    image = np.transpose(image, (1, 2, 0))  # 转换为 (H, W, 1)
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = torch.tensor(image, dtype=torch.float32)
                    self.images.append(image)
                    self.labels.append(label)
                    label_counts[label] += 1

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        for label, count in label_counts.items():
            print(f"Label {label}: {count} images loaded.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def stratified_split(dataset, train_ratio=0.8):
    labels = dataset.labels.numpy()  # 将标签转为 NumPy 数组
    train_indices = []
    val_indices = []
    
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        split = int(train_ratio * len(label_indices))
        train_indices.extend(label_indices[:split])
        val_indices.extend(label_indices[split:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 打印训练集和验证集中每个标签的数量
    train_labels = dataset.labels[train_indices].numpy()
    val_labels = dataset.labels[val_indices].numpy()
    
    print("Training set label distribution:")
    for label in np.unique(train_labels):
        print(f"Label {label}: {np.sum(train_labels == label)}")
    
    print("Validation set label distribution:")
    for label in np.unique(val_labels):
        print(f"Label {label}: {np.sum(val_labels == label)}")
    
    return train_dataset, val_dataset

def get_dataloader(image_dirs, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=4, use_cuda=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.0625, 0.0625), scale=None, shear=None),
        transforms.Resize((config.INPUT_SIZE_HIGH, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    
    dataset = GameDataset(image_dirs, transform)
    
    train_dataset, val_dataset = stratified_split(dataset)

    if use_cuda:
        train_dataset.dataset.images = train_dataset.dataset.images.cuda()
        train_dataset.dataset.labels = train_dataset.dataset.labels.cuda()
        val_dataset.dataset.images = val_dataset.dataset.images.cuda()
        val_dataset.dataset.labels = val_dataset.dataset.labels.cuda()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
