"""
PyTorch Dataset and DataLoader 示例

这个示例演示如何使用自定义 Dataset 和 DataLoader：
- 定义一个继承自 torch.utils.data.Dataset 的类
- 实现 __len__ 和 __getitem__
- 用 DataLoader 将数据封装成小批量并自动打乱
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """自定义数据集示例。"""

    def __init__(self, data, labels):
        """初始化数据和标签。"""
        self.data = data
        self.labels = labels

    def __len__(self):
        """返回数据总条数。"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引返回单条数据和对应标签。"""
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        out = self.linear(x)
        return out
    
def train_step(model, data, labels):
    model.train()
    outputs = model(data)
    model.optimizer.zero_grad()  # 清零梯度
    loss = model.loss_fn(outputs, labels)
    loss.backward()
    model.optimizer.step()
    print('Loss:', loss.item())
    

def main():
    # 假设我们有 30 个样本，每个样本 10 个特征
    data = torch.arange(3000, dtype=torch.float32).reshape(300, 10)  # 30 samples, 10 feature each
    labels = torch.tensor([0, 1, 2, 1, 2, 2, 0, 1, 0, 2] * 30, dtype=torch.long)  # 30 labels

    # 1. 创建 Dataset
    dataset = SimpleDataset(data, labels)
    print('Dataset size:', len(dataset))

    # 2. 创建 DataLoader：batch_size=4，shuffle=True
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 3. 迭代 DataLoader
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        print(f'Batch {batch_idx}')
        print(' batch_data shape:', batch_data.shape)
        print(' batch_labels shape:', batch_labels.shape)
        print(' batch_data:', batch_data)
        print(' batch_labels:', batch_labels)
        print('-' * 40)

        # 这里可以把 batch_data 放入模型进行训练 / 推理
        # outputs = model(batch_data)
        # loss = criterion(outputs, batch_labels)

    layer = model(input_dim=10, output_dim=3)
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        train_step(layer, batch_data, batch_labels)

    test_data = torch.tensor([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
    layer.eval()
    with torch.no_grad():
        test_outputs = layer(test_data)
        print('Test outputs:', test_outputs)


if __name__ == '__main__':
    
    main()
