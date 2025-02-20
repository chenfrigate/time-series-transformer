import torch
import pandas as pd
from models.transformer import TimeSeriesTransformer

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=5, model_dim=64, num_heads=4, num_layers=2).to(device)
model.load_state_dict(torch.load("models/transformer_model.pth"))
model.eval()

# 读取测试数据
df = pd.read_csv("data/test.csv")
x_test = torch.tensor(df.values, dtype=torch.float32).to(device)

# 预测
with torch.no_grad():
    y_pred = model(x_test).cpu().numpy()

# 保存预测结果
df["predictions"] = y_pred
df.to_csv("data/predictions.csv", index=False)
print("✅ 预测完成，结果已保存到 data/predictions.csv")

