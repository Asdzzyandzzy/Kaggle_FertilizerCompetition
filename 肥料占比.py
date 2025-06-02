import pandas as pd

# 1. 读取数据
data = pd.read_csv("data/train.csv")  # 替换为你的文件名

# 2. 统计作物+土壤的肥料分布
result = data.groupby(['Crop Type', 'Soil Type', 'Fertilizer Name']).size().reset_index(name='Count')

# 3. 计算占比
total_per_group = result.groupby(['Crop Type', 'Soil Type'])['Count'].transform('sum')
result['Percentage'] = result['Count'] / total_per_group * 100

# 4. 输出结果
print("=== 每种作物+土壤对应的肥料分布（占比） ===")
for (crop, soil) in result[['Crop Type', 'Soil Type']].drop_duplicates().values:
    print(f"\n作物: {crop}, 土壤: {soil}")
    sub = result[(result['Crop Type'] == crop) & (result['Soil Type'] == soil)]
    for _, row in sub.iterrows():
        print(f"  肥料: {row['Fertilizer Name']} - 占比: {row['Percentage']:.2f}%")

# 可选：保存结果为 CSV
result.to_csv('crop_soil_fertilizer_distribution.csv', index=False)
