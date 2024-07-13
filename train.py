import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

from dataloader import get_dataloader
from model import get_model
import config

def compute_mpa50(outputs, labels):
    preds = torch.argmax(outputs, dim=1)  # 获取预测类别
    correct = (preds == labels).float()
    return correct.sum() / correct.numel()

def compute_mpa50_95(outputs, labels):
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for thresh in thresholds:
        preds = torch.argmax(outputs, dim=1)  # 获取预测类别
        correct = (preds == labels).float()
        aps.append(correct.sum() / correct.numel())
    return sum(aps) / len(aps)

def train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate_one_epoch(model, val_loader, criterion, device, log):
    model.eval()
    val_running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    mpa50 = compute_mpa50(all_outputs, all_labels).item()
    mpa50_95 = compute_mpa50_95(all_outputs, all_labels).item()

    # 记录详细信息到日志文件
    preds = torch.argmax(all_outputs, dim=1)
    log.write("Detailed prediction vs. labels:\n")
    for i in range(len(preds)):
        log.write(f"Prediction: {preds[i].item()}, Label: {all_labels[i].item()}, Outputs: {all_outputs[i].cpu().numpy()}\n")
    
    return val_epoch_loss, mpa50, mpa50_95

def warmup_lr_scheduler(optimizer, warmup_percent, total_steps, base_lr):
    warmup_steps = int(warmup_percent * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main(args):
    # 设置是否使用 CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # 获取数据加载器
    image_dirs = [config.GOLDFLASH_IMAGE_DIR, config.RED_FLASH_IMAGE_DIR, config.NO_FLASH_IMAGE_DIR]
    train_loader, val_loader = get_dataloader(image_dirs, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=0, use_cuda=use_cuda)
    
    # 获取模型
    model_name = args.model if args.model else 'cnn'
    model = get_model(model_name=model_name, input_height=config.INPUT_SIZE_HIGH, input_width=config.INPUT_SIZE, num_classes=3).to(device)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_mpa50 = 0.0
    best_mpa50_95 = 0.0
    last_model_path = 'model/last_model.pth'
    best_model_path = 'model/best_model.pth'
    best_mpa50_model_path = 'model/best_mpa50_model.pth'
    log_file = 'training_log.txt'
    
    # 计算总的训练步骤数
    total_steps = config.NUM_EPOCHS * len(train_loader)
    scheduler = warmup_lr_scheduler(optimizer, config.WARMUP_PERCENT, total_steps, config.LEARNING_RATE)

    # 训练和验证模型
    with open(log_file, 'w') as log:
        log.write(f'Hyperparameters:\nLearning Rate: {config.LEARNING_RATE}\nBatch Size: {config.BATCH_SIZE}\n')
        log.write(f'Warmup Percent: {config.WARMUP_PERCENT}\n\n')
        log.write(f'Epoch, Training Loss, Validation Loss, mPA50, mPA50-95, Learning Rate\n')

        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Training Loss: {train_loss:.4f}')

            val_loss, mpa50, mpa50_95 = validate_one_epoch(model, val_loader, criterion, device, log)
            print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Validation Loss: {val_loss:.4f}, mPA50: {mpa50:.4f}, mPA50-95: {mpa50_95:.4f}')

            current_lr = optimizer.param_groups[0]['lr']
            log.write(f'{epoch+1}, {train_loss:.4f}, {val_loss:.4f}, {mpa50:.4f}, {mpa50_95:.4f}, {current_lr:.6f}\n')

            # 保存最后一个 epoch 的模型
            torch.save(model.state_dict(), last_model_path)

            # 如果当前 mPA50-95 更高，则保存最好的模型
            if mpa50_95 > best_mpa50_95:
                best_mpa50_95 = mpa50_95
                torch.save(model.state_dict(), best_model_path)
            
            # 如果当前 mPA50 更高，则保存最好的模型
            if mpa50 > best_mpa50:
                best_mpa50 = mpa50
                torch.save(model.state_dict(), best_mpa50_model_path)

    print('Finished Training')
    print(f'Last model saved to {last_model_path}')
    print(f'Best mPA50 model saved to {best_mpa50_model_path} with mPA50 {best_mpa50:.4f}')
    print(f'Best mPA50-95 model saved to {best_model_path} with mPA50-95 {best_mpa50_95:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--model', type=str, choices=['cnn', 'resnet', 'conformer'], default='cnn', help='Model type to use for training')
    args = parser.parse_args()
    main(args)
