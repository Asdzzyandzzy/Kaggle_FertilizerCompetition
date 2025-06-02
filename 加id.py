import pandas as pd

# 1. 读取你的数据
data = pd.read_csv("data/data_core.csv")  # 替换为你的文件名

# 2. 添加 id 列
data.insert(0, 'id', range(len(data)))

# 3. 保存新文件
data.to_csv('123.csv', index=False)

print("✅ 已成功生成包含 id 的数据文件：your_data_with_id.csv")
