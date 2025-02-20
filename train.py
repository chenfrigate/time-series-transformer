import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TimeSeriesTransformer
from utils.data_loader import load_dataset

# 训练参数
EPOCHS = 10
BATCH_SIZE = 32
SEQ_LENGTH = 10
LEARNING_RATE = 0.001

# 加载数据
train_loader = load_dataset("data/train.csv", SEQ_LENGTH, BATCH_SIZE)

# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=5, model_dim=64, num_heads=4, num_layers=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# 训练循环
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "models/transformer_model.pth")

