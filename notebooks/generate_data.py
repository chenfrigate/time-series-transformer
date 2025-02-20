
import pandas as pd
import numpy as np

# 设定随机种子，保证每次生成数据一致
np.random.seed(42)

# 数据集大小
num_train_samples = 1000  # 训练数据
num_test_samples = 200  # 测试数据
seq_length = 10  # 预测窗口大小

# 生成时间戳
timestamps_train = pd.date_range(start="2023-01-01", periods=num_train_samples, freq="H")
timestamps_test = pd.date_range(start="2023-02-01", periods=num_test_samples, freq="H")

# 生成特征数据
feature_1_train = np.random.randn(num_train_samples) * 10
feature_2_train = np.random.randn(num_train_samples) * 5
feature_3_train = np.random.randn(num_train_samples) * 3
feature_4_train = np.random.randn(num_train_samples) * 7  # 作为预测目标

feature_1_test = np.random.randn(num_test_samples) * 10
feature_2_test = np.random.randn(num_test_samples) * 5
feature_3_test = np.random.randn(num_test_samples) * 3
feature_4_test = np.random.randn(num_test_samples) * 7  # 作为预测目标

# 组合成 DataFrame
df_train = pd.DataFrame({
    "timestamp": timestamps_train,
    "feature_1": feature_1_train,
    "feature_2": feature_2_train,
    "feature_3": feature_3_train,
    "feature_4": feature_4_train  # 现在我们预测这个
})

df_test = pd.DataFrame({
    "timestamp": timestamps_test,
    "feature_1": feature_1_test,
    "feature_2": feature_2_test,
    "feature_3": feature_3_test,
    "feature_4": feature_4_test  # 现在我们预测这个
})

# 保存 CSV 文件
df_train.to_csv("../data/train.csv", index=False)
df_test.to_csv("../data/test.csv", index=False)

print("✅ 训练数据 train.csv 和 测试数据 test.csv 生成完成！")
