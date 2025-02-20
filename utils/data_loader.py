import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length):
        self.data = df.values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length, :]  # 取前 seq_length 个时间步作为输入
        y = self.data[index+self.seq_length, -1]  # 取下一个时间步作为预测目标
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_dataset(csv_path, seq_length=10, batch_size=32):
    df = pd.read_csv(csv_path)  # 读取 CSV 数据
    dataset = TimeSeriesDataset(df, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
