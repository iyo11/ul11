import os
import cv2
import torch
import numpy as np
import glob
import argparse
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


def run_inference(image_dir, output_dir, yolo_path, sam_path):
    # 1. 初始化模型
    print(f"Loading YOLO model from: {yolo_path}")
    yolo_model = YOLO(yolo_path)  # 加载您的 yolov11n.pt

    print("Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 2. 准备输出目录
    # Gaussian Grouping 默认去这就找 mask
    mask_dir = os.path.join(output_dir, "object_mask")
    os.makedirs(mask_dir, exist_ok=True)

    # 3. 遍历图片
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                         glob.glob(os.path.join(image_dir, "*.png")))

    print(f"Found {len(image_paths)} images. Processing...")

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]

        # --- A. 读取图片 ---
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- B. YOLOv11 检测 ---
        # classes=[39] 指定只检测瓶子 (COCO数据集里 bottle=39)
        # 如果你不加这个，桌子上的键盘、杯子、人手都会被分割进去，干扰3D建模
        results = yolo_model.predict(image_bgr, conf=0.25, classes=[39], verbose=False)

        # 如果没检测到，就跳过 (生成全黑 mask)
        if len(results[0].boxes) == 0:
            print(f"No bottle detected in {file_name}")  # 修改提示语
            empty_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            cv2.imwrite(os.path.join(mask_dir, f"{base_name}.png"), empty_mask)
            continue

        # 获取框 (x1, y1, x2, y2)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # --- C. SAM 分割 ---
        predictor.set_image(image_rgb)

        # 把 YOLO 的框喂给 SAM
        transformed_boxes = predictor.transform.apply_boxes_torch(
            torch.as_tensor(boxes, device=device), image_rgb.shape[:2]
        )

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # --- D. 保存结果 ---
        # 如果检测到多个部分，取并集 (只要是植物的一部分都算)
        final_mask = torch.any(masks, dim=0).squeeze().cpu().numpy().astype(np.uint8) * 255

        # 保存为单通道灰度图
        save_path = os.path.join(mask_dir, f"{base_name}.png")
        cv2.imwrite(save_path, final_mask)

        if len(image_paths) % 10 == 0:
            print(f"Processed: {file_name}")

    print("\n✅ 所有 Mask 生成完毕！保存在:", mask_dir)
    print("现在可以开始训练 Gaussian Grouping 了。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Convert.py 生成的数据目录 (例如 data/my_plant)")
    parser.add_argument("--yolo_pt", type=str, default="yolov11n.pt", help="您的 yolov11n.pt 文件路径")
    args = parser.parse_args()

    # 自动推断 image 目录
    img_dir = os.path.join(args.data_path, "images")
    if not os.path.exists(img_dir):
        print(f"错误: 找不到 images 目录: {img_dir}")
        print("请先运行 convert.py！")
    else:
        run_inference(img_dir, args.data_path, args.yolo_pt, "pts/sam_vit_h_4b8939.pth")