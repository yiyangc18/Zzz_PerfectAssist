import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes=3):
        super(ClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout = nn.Dropout(0.5)

        # 动态计算全连接层的输入大小
        dummy_input = torch.zeros(1, 3, input_height, input_width)
        flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x

    def _get_flattened_size(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x.view(x.size(0), -1).size(1)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=3, expansion_factor=2, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        # 逐点卷积，扩展通道数
        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model * expansion_factor, kernel_size=1)
        self.glu_activation = nn.GLU(dim=1)
        # 深度卷积，保留通道数
        self.depthwise_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, groups=d_model, padding=kernel_size // 2)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.swish_activation = nn.SiLU()
        # 逐点卷积，恢复通道数
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.glu_activation(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.d_model = d_model

        # 前馈网络，注意力前
        self.ff_pre_self_attn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        # 卷积模块
        self.conv_module = ConvolutionModule(d_model=d_model, dropout=dropout)
        # 前馈网络，注意力后
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # LayerNorm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 转置 batch 和 sequence 维度
        x = x.transpose(0, 1)
        residual = x

        # 前馈网络之前的 LayerNorm
        x = self.norm1(x)
        x = self.ff_pre_self_attn(x)
        x += residual

        residual = x
        # 注意力之前的 LayerNorm
        x = self.norm2(x)
        x, _ = self.self_attn(x, x, x)
        x += residual

        residual = x
        # 卷积之前的转置
        x = x.permute(1, 2, 0)
        x = self.conv_module(x)
        x = x.permute(2, 0, 1)

        residual = x
        # 卷积之后的 LayerNorm
        x = self.norm3(x)
        x = self.feed_forward(x)
        x += residual

        return x

class ConformerClassifier(nn.Module):
    def __init__(self, input_height, input_width, num_blocks, d_model=128, num_classes=3, dropout=0.1):
        super(ConformerClassifier, self).__init__()

        # 卷积层和批量归一化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 计算卷积输出大小
        dummy_input = torch.zeros(1, 3, input_height, input_width)
        conv_output_size = self._get_conv_output_size(dummy_input)
        self.d_model = conv_output_size

        # 多个 ConformerBlock
        self.conformer_blocks = nn.ModuleList(
            [ConformerBlock(d_model=self.d_model, dropout=dropout) for _ in range(num_blocks)]
        )

        # 最后的 LayerNorm 和全连接层
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_layer = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # 卷积和池化层
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平并转置
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, -1, height * width).permute(0, 2, 1)

        # 通过多个 ConformerBlock
        for block in self.conformer_blocks:
            x = block(x)

        # 最后的 LayerNorm 和全连接层
        x = self.final_norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.output_layer(x)
        return x

    def _get_conv_output_size(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        batch_size, _, height, width = x.size()
        return x.view(batch_size, -1, height * width).permute(0, 2, 1).size(2)


def get_model(model_name='cnn', input_height=412, input_width=234, num_classes=3):
    if model_name == 'cnn':
        return ClassificationCNN(input_height=input_height, input_width=input_width, num_classes=num_classes)
    elif model_name == 'conformer':
        return ConformerClassifier(input_height=input_height, input_width=input_width, num_blocks=2, d_model=16, num_classes=3, dropout=0.4)
    else:
        raise ValueError(f"Unknown model name '{model_name}'")

if __name__ == "__main__":
    from dataloader import get_dataloader
    import config
    import torch
    import torch.nn as nn
    import numpy as np

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dirs = [config.GOLDFLASH_IMAGE_DIR, config.RED_FLASH_IMAGE_DIR, config.NO_FLASH_IMAGE_DIR]
    train_loader, val_loader = get_dataloader(image_dirs, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=0, use_cuda=use_cuda)

    def inspect_model_output(model, data_loader, device, num_samples=3):
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                if i >= num_samples:
                    break
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                print(f"Batch {i+1}:")
                print(f"Outputs: {outputs}")
                print(f"Predictions: {torch.argmax(outputs, dim=1)}")
                print(f"Labels: {labels}")

    for model_name in ['conformer', 'cnn']:
        print(f"Testing model: {model_name}")
        model = get_model(model_name=model_name, input_height=config.INPUT_SIZE_HIGH, input_width=config.INPUT_SIZE, num_classes=3).to(device)
#         print(model)
        inspect_model_output(model, val_loader, device)
