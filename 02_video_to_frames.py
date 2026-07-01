# -*- coding: utf-8 -*-
# 02_video_to_frames.py - 批量拆帧（含自有数据集dataset_self，PIL保存版）
import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image  # ✅ 导入PIL用于保存帧

# ===================== 1. 全局路径配置（和01_config.py完全一致） =====================
DATASET_ROOT = {
    "ShanghaiTech": r"D:\毕业设计\dataset\ShanghaiTech_Video_Frame",
    "UCSD": r"D:\毕业设计\dataset\UCSD_Anomaly_Dataset",
    "UMN": r"D:\毕业设计\dataset\UMN",
    "dataset_self": r"D:\毕业设计\dataset\dataset_self"
}

# 拆帧参数
FRAME_SAVE_FORMAT = "frame_{:04d}.jpg"  # 帧命名格式
FRAME_SAVE_DIR_NAME = "frames"          # 帧保存文件夹名
VIDEO_SUFFIX = [".avi", ".mp4"]         # 支持的视频格式

# ===================== 2. 通用拆帧函数（PIL保存，解决OpenCV保存失败） =====================
def video_to_frames(video_path, save_dir):
    """
    单个视频拆帧（用PIL保存，兼容中文/长路径，解决OpenCV保存失败）
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🎬 开始处理视频：{video_path.name}")
    print(f"📂 帧保存路径：{save_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}（请检查视频是否损坏/路径是否正确）")
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 成功打开视频，总帧数：{total_frames}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = save_dir / FRAME_SAVE_FORMAT.format(frame_count + 1)
        try:
            # OpenCV BGR → PIL RGB 转换，用PIL保存
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.save(str(frame_filename), "JPEG", quality=95)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"📸 已保存 {frame_count}/{total_frames} 帧", end="\r")
        except Exception as e:
            print(f"\n❌ 保存帧失败：{frame_filename}，错误：{str(e)}")
            continue

    cap.release()
    print(f"\n✅ 视频 {video_path.name} 拆帧完成！共保存 {frame_count} 帧")
    return frame_count

# ===================== 3. 分数据集拆帧（原逻辑完全保留） =====================
def process_shanghaitech():
    print("\n" + "="*50)
    print("开始处理 ShanghaiTech 数据集")
    print("="*50)
    
    video_root = Path(DATASET_ROOT["ShanghaiTech"]) / "Test"
    if not video_root.exists():
        print(f"❌ ShanghaiTech视频根路径不存在：{video_root}")
        return
    
    video_files = []
    for suffix in VIDEO_SUFFIX:
        video_files.extend(video_root.glob(f"*{suffix}"))
    
    if not video_files:
        print(f"⚠️ ShanghaiTech {video_root} 下未找到视频文件")
        return
    
    total_frames = 0
    for video in tqdm(video_files, desc="ShanghaiTech拆帧进度"):
        save_dir = video_root / f"{video.stem}_{FRAME_SAVE_DIR_NAME}"
        total_frames += video_to_frames(video, save_dir)
    
    print(f"\n✅ ShanghaiTech 数据集处理完成：共拆{len(video_files)}个视频，{total_frames}帧")

def process_ucsd():
    print("\n" + "="*50)
    print("开始处理 UCSD 数据集（已有.tif帧，跳过拆帧）")
    print("="*50)
    print("✅ UCSD 原始数据已是.tif帧，无需拆帧，直接跳过")

def process_umn():
    print("\n" + "="*50)
    print("开始处理 UMN 数据集")
    print("="*50)
    
    umn_root = Path(DATASET_ROOT["UMN"])
    scene_dirs = list(umn_root.glob("Scene_*"))
    if not scene_dirs:
        print(f"❌ UMN 下未找到 Scene_* 文件夹：{umn_root}")
        return
    
    total_frames = 0
    for scene in tqdm(scene_dirs, desc="UMN拆帧进度"):
        video_path = scene / "1.mp4"
        if not video_path.exists():
            print(f"⚠️ UMN {scene.name} 下未找到1.mp4，跳过")
            continue
        
        save_dir = scene / FRAME_SAVE_DIR_NAME
        total_frames += video_to_frames(video_path, save_dir)
    
    print(f"\n✅ UMN 数据集处理完成：共拆{len(scene_dirs)}个视频，{total_frames}帧")

# ===================== 4. 【核心新增】处理自有数据集 dataset_self =====================
def process_dataset_self():
    print("\n" + "="*50)
    print("开始处理 自有数据集 dataset_self")
    print("="*50)
    
    self_root = Path(DATASET_ROOT["dataset_self"])
    if not self_root.exists():
        print(f"❌ 自有数据集路径不存在：{self_root}")
        return
    
    # 遍历dataset_self下所有mp4/avi视频
    video_files = []
    for suffix in VIDEO_SUFFIX:
        video_files.extend(self_root.glob(f"*{suffix}"))
    
    if not video_files:
        print(f"⚠️ dataset_self 下未找到视频文件，请检查路径/格式")
        return
    
    total_frames = 0
    # 给每个视频单独建frames文件夹，避免帧混在一起，结构清晰
    for video in tqdm(video_files, desc="dataset_self拆帧进度"):
        save_dir = self_root / f"{video.stem}_{FRAME_SAVE_DIR_NAME}"
        total_frames += video_to_frames(video, save_dir)
    
    print(f"\n✅ 自有数据集 dataset_self 处理完成：共拆{len(video_files)}个视频，{total_frames}帧")

# ===================== 5. 主函数（新增调用process_dataset_self） =====================
def main():
    # 检查依赖
    try:
        import cv2
        from PIL import Image
    except ImportError as e:
        print(f"⚠️ 缺少依赖：{e}，正在自动安装...")
        os.system("pip install opencv-python pillow")
        import cv2
        from PIL import Image
    
    # 按顺序处理数据集（最后加自有数据集，不影响原流程）
    process_shanghaitech()
    process_ucsd()  # UCSD已有帧，直接跳过
    process_umn()
    process_dataset_self()  # ✅ 新增：处理自己的dataset_self数据集
    
    print("\n" + "="*50)
    print("02号程序 - 拆帧任务结束！")
    print("下一步：运行03_dynamic_features.py提取动态特征")
    print("="*50)

if __name__ == "__main__":
    main()