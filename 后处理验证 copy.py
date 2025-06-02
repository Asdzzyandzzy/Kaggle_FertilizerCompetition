import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练数据
train = pd.read_csv("data/train.csv")

# 定义要验证的业务规则
def evaluate_business_rules(df):
    """
    在训练数据上评估业务规则的有效性
    返回两个规则的分析结果
    """
    results = {}
    
    # 规则1: 高温甘蔗 - 应该使用28-28
    sugarcane_high_temp = df[(df["Crop Type"] == "Sugarcane") & (df["Temparature"] > 35)]
    
    if not sugarcane_high_temp.empty:
        rule1_counts = sugarcane_high_temp["Fertilizer Name"].value_counts()
        rule1_percentage = rule1_counts.get("28-28", 0) / len(sugarcane_high_temp) * 100
        
        results["rule1"] = {
            "condition": "Sugarcane & Temp > 35",
            "sample_size": len(sugarcane_high_temp),
            "most_common": rule1_counts.idxmax() if not rule1_counts.empty else None,
            "28-28_percentage": rule1_percentage,
            "fertilizer_distribution": rule1_counts
        }
    else:
        results["rule1"] = {
            "condition": "Sugarcane & Temp > 35",
            "sample_size": 0,
            "message": "No samples in training data"
        }
    
    # 规则2: 低磷水稻 - 应该使用DAP
    paddy_low_phosphorous = df[(df["Crop Type"] == "Paddy") & (df["Phosphorous"] < 15)]
    
    if not paddy_low_phosphorous.empty:
        rule2_counts = paddy_low_phosphorous["Fertilizer Name"].value_counts()
        rule2_percentage = rule2_counts.get("DAP", 0) / len(paddy_low_phosphorous) * 100
        
        results["rule2"] = {
            "condition": "Paddy & Phosphorous < 15",
            "sample_size": len(paddy_low_phosphorous),
            "most_common": rule2_counts.idxmax() if not rule2_counts.empty else None,
            "DAP_percentage": rule2_percentage,
            "fertilizer_distribution": rule2_counts
        }
    else:
        results["rule2"] = {
            "condition": "Paddy & Phosphorous < 15",
            "sample_size": 0,
            "message": "No samples in training data"
        }
    
    return results

# 在训练数据上评估规则
rule_results = evaluate_business_rules(train)

# 打印结果
print("业务规则验证结果:\n")
for rule, data in rule_results.items():
    print(f"=== {rule.upper()} ===")
    print(f"条件: {data['condition']}")
    print(f"样本数量: {data.get('sample_size', 0)}")
    
    if "message" in data:
        print(data["message"])
    else:
        print(f"最常见的肥料: {data['most_common']}")
        
        if rule == "rule1":
            print(f"28-28 使用比例: {data['28-28_percentage']:.2f}%")
        else:
            print(f"DAP 使用比例: {data['DAP_percentage']:.2f}%")
        
        print("\n肥料分布:")
        print(data["fertilizer_distribution"])
    
    print("\n" + "-"*50 + "\n")

# 可视化规则1的结果
if "rule1" in rule_results and rule_results["rule1"]["sample_size"] > 0:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=rule_results["rule1"]["fertilizer_distribution"].index,
        y=rule_results["rule1"]["fertilizer_distribution"].values
    )
    plt.title("肥料分布 - 甘蔗 & 温度 > 35")
    plt.xlabel("肥料类型")
    plt.ylabel("数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("rule1_validation.png")
    plt.show()

# 可视化规则2的结果
if "rule2" in rule_results and rule_results["rule2"]["sample_size"] > 0:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=rule_results["rule2"]["fertilizer_distribution"].index,
        y=rule_results["rule2"]["fertilizer_distribution"].values
    )
    plt.title("肥料分布 - 水稻 & 磷 < 15")
    plt.xlabel("肥料类型")
    plt.ylabel("数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("rule2_validation.png")
    plt.show()

# 扩展分析：查看不同条件下肥料使用的变化趋势
def analyze_trends(df):
    """分析肥料使用随温度和磷水平的变化趋势"""
    # 甘蔗肥料使用随温度变化
    sugarcane = df[df["Crop Type"] == "Sugarcane"]
    if not sugarcane.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Fertilizer Name", y="Temparature", data=sugarcane)
        plt.title("甘蔗肥料使用与温度关系")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("sugarcane_temp_trend.png")
        plt.show()
    
    # 水稻肥料使用随磷水平变化
    paddy = df[df["Crop Type"] == "Paddy"]
    if not paddy.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Fertilizer Name", y="Phosphorous", data=paddy)
        plt.title("水稻肥料使用与磷水平关系")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("paddy_phosphorous_trend.png")
        plt.show()

# 执行趋势分析
analyze_trends(train)

print("验证完成！图表已保存。")