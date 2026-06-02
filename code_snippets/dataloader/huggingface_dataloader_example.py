"""
Hugging Face datasets + PyTorch DataLoader 示例

这个示例演示如何使用 Hugging Face 的 datasets 库加载数据集，
并将其与 PyTorch DataLoader 结合使用。

需要安装：
  pip install datasets torch
"""

from datasets import load_dataset
from torch.utils.data import DataLoader


def main():
    # 1. 从 Hugging Face Hub 下载数据集
    #    这里使用 glue/mrpc 作为示例，也可以替换为其他公开数据集
    dataset = load_dataset('glue', 'mrpc', split='train')
    print('Dataset example count:', len(dataset))
    print('First example:', dataset[0])

    # 2. 选择需要的字段，并设置 PyTorch 格式
    #    这里将句子对文本和标签一同转换为字典格式
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['sentence1', 'sentence2', 'label']])
    dataset.set_format(type='torch', columns=['sentence1', 'sentence2', 'label'])

    # 3. DataLoader 迭代 Hugging Face Dataset
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch_idx, batch in enumerate(dataloader):
        print(f'Batch {batch_idx}')
        print(' sentence1:', batch['sentence1'][:2])
        print(' sentence2:', batch['sentence2'][:2])
        print(' label:', batch['label'][:2])
        print('batch keys:', list(batch.keys()))
        break

    # 4. 如果需要：将 Hugging Face Dataset 转成 transformers tokenizer 之后再送入模型
    #    这时可以使用 dataset.map(tokenize_fn, batched=True) 并 set_format(type='torch', columns=[...])


if __name__ == '__main__':
    main()
