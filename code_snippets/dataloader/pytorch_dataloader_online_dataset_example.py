"""
PyTorch DataLoader + 在线下载数据集示例

这个示例演示如何使用 torchvision.datasets 从网络下载数据集，
并通过 DataLoader 进行批量加载。

示例使用:
- MNIST 数据集
- torchvision.transforms 进行简单张量转换
- DataLoader 进行 shuffle、batch
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    # 1. 定义数据转换：将 PIL 图像转为 Tensor，并归一化到 [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. 下载 MNIST 训练集（如果本地已存在，则不会重复下载）
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    print('Train dataset size:', len(train_dataset))

    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 4. 迭代 DataLoader 并打印一个 batch 的形状
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f'Batch {batch_idx}')
        print(' images shape:', images.shape)
        print(' targets shape:', targets.shape)
        print(' targets:', targets[:10].tolist())
        break

    # 这里可以将 images 传入网络进行训练
    # outputs = model(images)
    # loss = criterion(outputs, targets)


if __name__ == '__main__':
    main()
