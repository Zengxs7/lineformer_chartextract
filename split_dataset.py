import os
import shutil
import random

# ========== 1. 基本配置 ==========
# 图像与CSV的源路径
img_dir = r"D:\ZXS\new_dataset\images"
csv_dir = r"D:\ZXS\new_dataset\labels"

# 输出路径
base_out = r"D:\ZXS\dataset2"

# 训练集比例
train_ratio = 0.8
random.seed(42)

# 创建输出文件夹结构
train_img = os.path.join(base_out, "train", "images")
train_csv = os.path.join(base_out, "train", "csvs")
val_img = os.path.join(base_out, "val", "images")
val_csv = os.path.join(base_out, "val", "csvs")

for p in [train_img, train_csv, val_img, val_csv]:
    os.makedirs(p, exist_ok=True)

# ========== 2. 获取所有样本 ==========
files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
random.shuffle(files)

split_index = int(len(files) * train_ratio)
train_files = files[:split_index]
val_files = files[split_index:]

# ========== 3. 定义复制函数 ==========
def copy_files(file_list, img_src_dir, csv_src_dir, img_dst_dir, csv_dst_dir):
    count = 0
    for f in file_list:
        csv_src = os.path.join(csv_src_dir, f)
        img_name = f.replace(".csv", ".png")
        img_src = os.path.join(img_src_dir, img_name)

        if os.path.exists(csv_src) and os.path.exists(img_src):
            shutil.copy2(img_src, os.path.join(img_dst_dir, img_name))
            shutil.copy2(csv_src, os.path.join(csv_dst_dir, f))
            count += 1
        else:
            print(f"⚠️ 未找到匹配文件: {f}")
    return count

# ========== 4. 开始划分 ==========
train_count = copy_files(train_files, img_dir, csv_dir, train_img, train_csv)
val_count = copy_files(val_files, img_dir, csv_dir, val_img, val_csv)

# ========== 5. 输出结果 ==========
print("✅ 数据划分完成：")
print(f"  训练集: {train_count} 条样本")
print(f"  验证集: {val_count} 条样本")
print(f"  共计: {train_count + val_count}")
