import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置部分 =================
LINE_WIDTH = 3                # 折线粗细
FALLBACK_PLOT_SIZE = 805      # 检测失败时，使用居中的 805×805 绘图区

# 数据路径（按你现在的 dataset2 结构）
TRAIN_CSV_DIR = r"D:\ZXS\dataset2\train\csvs"
TRAIN_IMG_DIR = r"D:\ZXS\dataset2\train\images"
VAL_CSV_DIR   = r"D:\ZXS\dataset2\val\csvs"
VAL_IMG_DIR   = r"D:\ZXS\dataset2\val\images"

SAVE_DIR = r"D:\ZXS\annotations_805"
os.makedirs(SAVE_DIR, exist_ok=True)
# ===========================================


def find_image(csv_name, img_dir):
    """在文件夹中根据 csv 名找到对应图像： spectrum_xxxxx_*.png"""
    base = os.path.splitext(csv_name)[0]
    for f in os.listdir(img_dir):
        if f.startswith(base) and f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(img_dir, f)
    return None


def detect_plot_area(img):
    """
    自动检测绘图区（坐标轴外框范围）。
    返回 (x1, y1, x2, y2)。若失败，返回居中的 805×805。
    """
    h, w = img.shape[:2]

    # 先尝试检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, wc, hc = cv2.boundingRect(c)
        # 经验阈值：大框且接近整图（你原来代码的思路）
        if 0.6 * w < wc < 0.98 * w and 0.6 * h < hc < 0.98 * h:
            return x, y, x + wc, y + hc

    # 失败则使用居中 805×805（或在边界受限时尽量取正方形）
    S = min(FALLBACK_PLOT_SIZE, w, h)
    x1 = (w - S) // 2
    y1 = (h - S) // 2
    return x1, y1, x1 + S, y1 + S


def make_coco(csv_dir, img_dir, json_path):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "spectrum"}]
    }

    ann_id = 1
    img_id = 1

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for file in tqdm(csv_files, desc=f"Processing {os.path.basename(json_path)}"):
        csv_path = os.path.join(csv_dir, file)
        img_path = find_image(file, img_dir)
        if not img_path:
            print(f"[⚠] 找不到图片: {file}")
            continue

        df = pd.read_csv(csv_path)
        if "wavelength" not in df.columns or "intensity" not in df.columns:
            print(f"[⚠] CSV列缺失: {file}")
            continue
        if df.empty:
            print(f"[⚠] 空CSV文件: {file}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[⚠] 无法读取图像: {img_path}")
            continue

        H, W = img.shape[:2]                    # ← 使用真实图像尺寸
        x1, y1, x2, y2 = detect_plot_area(img)
        plot_w, plot_h = x2 - x1, y2 - y1

        xmin, xmax = float(df["wavelength"].min()), float(df["wavelength"].max())
        ymin, ymax = float(df["intensity"].min()), float(df["intensity"].max())
        if xmax == xmin: xmax += 1e-6
        if ymax == ymin: ymax += 1e-6

        # 以真实图像大小创建空白 mask
        mask = np.zeros((H, W), dtype=np.uint8)
        pts = []

        # --- 将CSV坐标映射到绘图区 ---
        for x, y in zip(df["wavelength"], df["intensity"]):
            x_img = int(round(x1 + (x - xmin) / (xmax - xmin) * plot_w))
            y_img = int(round(y2 - (y - ymin) / (ymax - ymin) * plot_h))
            pts.append([x_img, y_img])

        if len(pts) < 2:
            print(f"[⚠] 点数过少: {file}")
            continue

        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], False, color=1, thickness=LINE_WIDTH)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"[⚠] 无有效mask: {file}")
            continue

        cnt = max(contours, key=cv2.contourArea)
        segmentation = [cnt.flatten().tolist()]
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = [float(x), float(y), float(w), float(h)]
        area = float(cv2.contourArea(cnt))

        # --- COCO: 用真实宽高 ---
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": int(W),
            "height": int(H)
        })

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })

        ann_id += 1
        img_id += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 已生成 COCO 文件: {json_path}")
    print(f"共 {len(coco['images'])} 张图, {len(coco['annotations'])} 个标注")


# ================= 主程序 =================
os.makedirs(SAVE_DIR, exist_ok=True)
make_coco(TRAIN_CSV_DIR, TRAIN_IMG_DIR, os.path.join(SAVE_DIR, "instances_train.json"))
make_coco(VAL_CSV_DIR,   VAL_IMG_DIR,   os.path.join(SAVE_DIR, "instances_val.json"))
