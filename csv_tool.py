import pandas as pd
from pathlib import Path

def count_csv_samples_with_anomaly_types(csv_path):
    """
    统计CSV样本量：
    1. 验证总行数（77820）
    2. 按数据集+异常类型分组统计（含正常/各异常类型）
    3. 输出结构化结果（适配论文表格）
    """
    # 1. 读取CSV（兼容utf-8/gbk编码）
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")
    
    # 2. 验证总行数
    total_rows = len(df)
    print(f"✅ CSV文件总行数：{total_rows}")
    if total_rows == 77820:
        print("✅ 总行数与预期一致，统计可靠！")
    else:
        print(f"⚠️  总行数不符（预期77820，实际{total_rows}），请核对文件")
    print("-" * 80)

    # 3. 定义核心异常类型（和你论文一致，可根据CSV实际标签调整）
    core_anomaly_types = [
        "奔跑/突然变向", "人群拥挤/推挤", "车辆闯入", 
        "逆行/掉头", "徘徊/急停", "其他异常"
    ]
    normal_type = "正常行走"  # CSV中正常样本的标签

    # 4. 按「数据集+异常类型」分组统计
    # 全量分组统计
    group_count = df.groupby(["dataset", "anomaly_type"]).size().reset_index(name="样本数")
    # 筛选核心异常类型+正常类型，确保结果完整
    target_types = [normal_type] + core_anomaly_types
    filtered_count = group_count[group_count["anomaly_type"].isin(target_types)]

    # 5. 整理成“数据集为行、异常类型为列”的透视表（直观展示）
    pivot_result = filtered_count.pivot(
        index="dataset", 
        columns="anomaly_type", 
        values="样本数"
    ).fillna(0).astype(int)  # 无数据的类型填0

    # 补充“总计”行/列
    pivot_result["该数据集样本总数"] = pivot_result.sum(axis=1)  # 列总计（每个数据集的总数）
    pivot_result.loc["所有数据集汇总"] = pivot_result.sum(axis=0)  # 行总计（所有数据集的各类型总数）

    # 6. 输出结果（适配论文展示）
    print("📊 各数据集-各异常类型样本量统计（直接复制到论文）：")
    print(pivot_result.to_string())  # 完整表格
    print("-" * 80)

    # 7. 额外输出表3-1所需的“正常/异常汇总”
    print("📋 表3-1专用：各数据集正常/异常样本汇总")
    normal_summary = df[df["anomaly_type"] == normal_type].groupby("dataset").size().reset_index(name="正常样本数")
    abnormal_summary = df[df["anomaly_type"] != normal_type].groupby("dataset").size().reset_index(name="异常样本数")
    summary = pd.merge(normal_summary, abnormal_summary, on="dataset", how="outer").fillna(0).astype(int)
    summary["表3-1显示格式"] = summary.apply(lambda x: f"{x['正常样本数']}/{x['异常样本数']}", axis=1)
    print(summary.to_string(index=False))

# ===================== 执行统计（仅需修改文件路径）=====================
csv_file_path = Path("D:\毕业设计\chapter3_results\dynamic_features_v2.csv")  # 替换为你的CSV实际路径
count_csv_samples_with_anomaly_types(csv_file_path)