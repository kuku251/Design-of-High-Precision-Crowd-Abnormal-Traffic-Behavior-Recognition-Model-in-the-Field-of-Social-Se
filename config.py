# -*- coding: utf-8 -*-
# 01_config.py - 导师最终要求版
# 自有数据集：dataset_self | 异常行为：5类（奔跑/徘徊/人群拥挤/车辆闯入/跌倒）+ 正常行走
import os
import pandas as pd
from pathlib import Path
import cv2
import json

# ===================== 1. 全局路径配置（统一使用 dataset_self 作为自有数据集） =====================
DATASET_ROOT = {
    "ShanghaiTech": r"D:\毕业设计\dataset\ShanghaiTech_Video_Frame",
    "UCSD": r"D:\毕业设计\dataset\UCSD_Anomaly_Dataset",
    "UMN": r"D:\毕业设计\dataset\UMN",
    "dataset_self": r"D:\毕业设计\dataset\dataset_self"  # 你的自有数据集路径
}

# ===================== 2. 导师强制要求：6类分类（无其他异常/逆行） =====================
ANNOTATION_TYPES = [
    "奔跑", "徘徊", "人群拥挤", "车辆闯入", "跌倒", "正常行走"
]

# 最终合并标注文件路径
FINAL_ANNOTATION_CSV = r"D:\毕业设计\dataset\all_dataset_annotations.csv"

# ===================== 3. ShanghaiTech 数据集标注（精简为5类异常） =====================
def annotate_shanghaitech():
    json_path = os.path.join(DATASET_ROOT["ShanghaiTech"], "test.json")
    video_type_map = {
        "04_0011.avi": "奔跑",
        "04_0012.avi": "奔跑",
        "04_0046.avi": "徘徊",
        "05_0017.avi": "徘徊",
        "05_0019.avi": "人群拥挤",
        "05_0023.avi": "跌倒",
        "06_0144.avi": "车辆闯入",
        "06_0153.avi": "车辆闯入",
        "07_0006.avi": "跌倒",
        # "07_0008.avi": "其他异常",
        "07_0048.avi": "奔跑",
        "08_0058.avi": "奔跑",
        "08_0077.avi": "车辆闯入",
        # "08_0080.avi": "徘徊",
        # "08_0157.avi": "奔跑",
        "10_0038.avi": "车辆闯入",
        "10_0042.avi": "车辆闯入",
        "12_0142.avi": "车辆闯入",
        "12_0151.avi": "车辆闯入",
        "12_0173.avi": "车辆闯入",
        "12_0174.avi": "车辆闯入",
        "default": "正常行走"
    }

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        print(f"❌ 找不到ShanghaiTech标注文件")
        return []

    annotation_list = []
    default_type = video_type_map["default"]
    for video_name in data.keys():
        annot_type = video_type_map.get(video_name, default_type)
        frame_labels = data[video_name]
        scene_num = video_name[:2]

        if annot_type == "正常行走":
            annotation_list.append({
                "dataset": "ShanghaiTech",
                "video_name": video_name,
                "anomaly_type": annot_type,
                "start_frame": 1,
                "end_frame": len(frame_labels),
                "scene": f"校园场景{scene_num}"
            })
        else:
            start = None
            for idx, val in enumerate(frame_labels):
                frame_num = idx + 1
                if val == 1 and start is None:
                    start = frame_num
                elif val == 0 and start is not None:
                    annotation_list.append({
                        "dataset": "ShanghaiTech",
                        "video_name": video_name,
                        "anomaly_type": annot_type,
                        "start_frame": start,
                        "end_frame": frame_num - 1,
                        "scene": f"校园场景{scene_num}"
                    })
                    start = None
            if start is not None:
                annotation_list.append({
                    "dataset": "ShanghaiTech",
                    "video_name": video_name,
                    "anomaly_type": annot_type,
                    "start_frame": start,
                    "end_frame": len(frame_labels),
                    "scene": f"校园场景{scene_num}"
                })
    return annotation_list

# ===================== 4. UCSD 数据集标注 =====================
def annotate_ucsd():
    annotation_list = []
    ucsd_root = Path(DATASET_ROOT["UCSD"])

    ped1_abnormal_type_map = {
        "Test003": "车辆闯入",
        "Test004": "正常行走", 
        "Test014": "车辆闯入",
        "Test018": "正常行走",
        "Test019": "车辆闯入",
        "Test021": "车辆闯入",
        "Test022": "正常行走",
        "Test023": "正常行走",
        "Test024": "车辆闯入",
        "Test032": "车辆闯入"
    }

    ped2_abnormal_type_map = {
        "Test001": "车辆闯入",
        "Test002": "车辆闯入",
        "Test003": "车辆闯入",
        "Test004": "车辆闯入",
        "Test005": "车辆闯入",
        "Test006": "车辆闯入",
        "Test007": "车辆闯入",
        "Test008": "车辆闯入",
        "Test009": "车辆闯入",
        "Test010": "车辆闯入",
        "Test011": "车辆闯入",
        "Test012": "正常行走"
    }

    for ped_version in ["UCSDped1", "UCSDped2"]:
        ped_path = ucsd_root / ped_version
        if not ped_path.exists(): continue

        train_dir = ped_path / "Train"
        if train_dir.exists():
            for train_subdir in train_dir.glob("Train*"):
                frame_count = len(list(train_subdir.glob("*.tif")))
                annotation_list.append({
                    "dataset": "UCSD",
                    "video_name": f"{ped_version}_{train_subdir.name}.avi",
                    "anomaly_type": "正常行走",
                    "start_frame": 1,
                    "end_frame": frame_count,
                    "scene": f"UCSD {ped_version} 训练集"
                })

        test_dir = ped_path / "Test"
        if test_dir.exists():
            test_subdirs = [d for d in test_dir.glob("Test*") if "_gt" not in d.name]
            gt_dirs = [d.name.replace("_gt", "") for d in test_dir.glob("Test*_gt")]

            for test_subdir in test_subdirs:
                test_name = test_subdir.name
                frame_count = len(list(test_subdir.glob("*.tif")))
                if ped_version == "UCSDped1":
                    annot_type = ped1_abnormal_type_map[test_name] if test_name in gt_dirs else "正常行走"
                else:
                    annot_type = ped2_abnormal_type_map[test_name]

                annotation_list.append({
                    "dataset": "UCSD",
                    "video_name": f"{ped_version}_{test_name}.avi",
                    "anomaly_type": annot_type,
                    "start_frame": 1,
                    "end_frame": frame_count,
                    "scene": f"UCSD {ped_version} 测试集"
                })
    return annotation_list

# ===================== 5. UMN 数据集标注 =====================
def annotate_umn():
    annotation_list = []
    umn_root = Path(DATASET_ROOT["UMN"])

    for scene_dir in umn_root.glob("Scene_*"):
        video_path = scene_dir / "1.mp4"
        if not video_path.exists(): continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        scene_name = scene_dir.name

        annotation_list.append({
            "dataset": "UMN",
            "video_name": f"{scene_name}_1.mp4",
            "anomaly_type": "奔跑",
            "start_frame": 1,
            "end_frame": frame_count,
            "scene": f"公共场所恐慌场景{scene_dir.name[-1]}"
        })
    return annotation_list

# ===================== 6. 自有数据集标注（dataset_self） =====================
def annotate_dataset_self():
    """自有数据集自动标注：根据视频文件名识别类别"""
    annotation_list = []
    self_root = Path(DATASET_ROOT["dataset_self"])
    if not self_root.exists():
        print("⚠️ 自有数据集路径不存在，跳过")
        return annotation_list

    # 支持 mp4/avi 格式
    video_files = list(self_root.glob("*.mp4")) + list(self_root.glob("*.avi"))
    for video_path in video_files:
        video_name = video_path.name
        # 自动从文件名匹配类别
        if "奔跑" in video_name:
            annot_type = "奔跑"
        elif "徘徊" in video_name:
            annot_type = "徘徊"
        elif "拥挤" in video_name:
            annot_type = "人群拥挤"
        elif "车辆" in video_name:
            annot_type = "车辆闯入"
        elif "跌倒" in video_name:
            annot_type = "跌倒"
        else:
            annot_type = "正常行走"

        # 获取视频总帧数
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 100
        cap.release()

        annotation_list.append({
            "dataset": "dataset_self",
            "video_name": video_name,
            "anomaly_type": annot_type,
            "start_frame": 1,
            "end_frame": frame_count,
            "scene": "自有采集场景"
        })
    print(f"✅ 自有数据集标注完成：共{len(annotation_list)}个视频")
    return annotation_list

# ===================== 7. 合并所有数据集 =====================
def main():
    sh_anno = annotate_shanghaitech()
    ucsd_anno = annotate_ucsd()
    umn_anno = annotate_umn()
    self_anno = annotate_dataset_self()  # 加载自有数据

    all_anno = sh_anno + ucsd_anno + umn_anno + self_anno
    if not all_anno:
        print("❌ 未生成任何标注！")
        return

    df = pd.DataFrame(all_anno)
    # 严格过滤：只保留导师要求的6类
    df = df[df["anomaly_type"].isin(ANNOTATION_TYPES)]

    df.to_csv(FINAL_ANNOTATION_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 全数据集合并完成（包含自有采集数据）")
    print(f"📊 最终类别统计：")
    print(df.groupby(["dataset", "anomaly_type"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()