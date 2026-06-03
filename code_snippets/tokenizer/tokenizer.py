from transformers import AutoTokenizer
from datasets import load_dataset

# 1. 加载对应的 Tokenizer（以经典的 BERT 为例，如果是 LLaMA 就换成 LLaMA 的路径）
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 2. 准备数据集
dataset = load_dataset('glue', 'mrpc', split='train')

# 3. 【核心】编写分词处理函数xxxxxxxxx
def tokenize_function(examples):
    # 句子对任务（如 MRPC）：直接把 sentence1 和 sentence2 两个列表传进去
    return tokenizer(
        examples['sentence1'], 
        examples['sentence2'], 
        truncation=True,             # 超过最大长度就截断
        padding=False,               # 👈 此时先不加 padding，留给 DataLoader 动态加，省显存！
        max_length=128               # 限制最大长度
    )

# 4. 调用 map 函数进行全量多进程分词
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,        # 必须为 True，表示一批批处理，速度极快
    num_proc=0           #
)

print(tokenized_dataset[0])