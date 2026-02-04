import os
import cv2
import numpy as np
import glob
import argparse
from ultralytics import YOLO


def run_inference_fast(image_dir, output_dir, yolo_seg_path):
    # --- 1. 加载模型 ---
    print(f"🔄 正在加载模型: {yolo_seg_path}")

    try:
        # YOLO 类会自动处理：
        # 1. 如果路径存在，直接加载
        # 2. 如果路径不存在但名字是标准名(如 yolo11n-seg.pt)，会自动下载
        model = YOLO(yolo_seg_path)
    except Exception as e:
        print(f"\n❌ 错误: 无法加载模型 -> {yolo_seg_path}")
        print(f"系统报错信息: {e}")
        print("💡 提示: 请检查路径是否正确，或者手动下载放入该路径。")
        print("下载地址: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt")
        return

    # --- 2. 准备输出目录 ---
    mask_dir = os.path.join(output_dir, "object_mask_yolo")
    os.makedirs(mask_dir, exist_ok=True)

    # --- 3. 遍历图片 ---
    # 支持 jpg, png, jpeg
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpeg"))
    )

    if len(image_paths) == 0:
        print(f"❌ 在 {image_dir} 下没有找到图片！")
        return

    print(f"📂 发现 {len(image_paths)} 张图片，开始 YOLO-Seg 极速处理...")

    for i, img_path in enumerate(image_paths):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]

        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 无法读取图片: {file_name}，跳过。")
            continue

        h, w = image.shape[:2]

        # --- YOLO 推理 ---
        # classes=[39] -> 指定瓶子 (Bottle)
        # retina_masks=True -> 开启高分辨率 Mask
        results = model.predict(image, conf=0.25, classes=[39], retina_masks=True, verbose=False)

        # 初始化全黑 Mask
        final_mask = np.zeros((h, w), dtype=np.uint8)

        # 检查是否检测到 Mask
        if results[0].masks is not None:
            # 获取 Mask 数据
            masks_data = results[0].masks.data.cpu().numpy()

            for mask_tensor in masks_data:
                # 确保尺寸匹配
                if mask_tensor.shape != (h, w):
                    mask_tensor = cv2.resize(mask_tensor, (w, h))

                # 二值化 (0-1 float 转 0/255 uint8)
                mask_binary = (mask_tensor > 0.5).astype(np.uint8) * 255

                # 合并多个瓶子的 Mask
                final_mask = cv2.bitwise_or(final_mask, mask_binary)
        else:
            # 如果没检测到，保持全黑，不打印刷屏，静默处理
            pass

            # 保存 Mask
        save_path = os.path.join(mask_dir, f"{base_name}.png")
        cv2.imwrite(save_path, final_mask)

        # 进度条
        if (i + 1) % 10 == 0:
            print(f"✅ 已处理 {i + 1}/{len(image_paths)} 张: {file_name}")

    print(f"\n🎉 处理完毕！Mask 已保存在: {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11-Seg 快速生成 Mask 工具")

    # 必须参数：数据路径
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据根目录 (包含 images 文件夹)")

    # 可选参数：权重路径 (默认 yolo11n-seg.pt)
    # 你可以在命令行用 --yolo_pt "F:/models/my_model.pt" 来指定
    parser.add_argument("--yolo_pt", type=str, default="yolo11n-seg.pt",
                        help="YOLO 分割模型路径 (例如: yolo11n-seg.pt 或 F:/weights/best.pt)")

    args = parser.parse_args()

    # 路径检查
    img_dir = os.path.join(args.data_path, "images")
    if os.path.exists(img_dir):
        # 把解析到的 yolo_pt 参数传进去
        run_inference_fast(img_dir, args.data_path, args.yolo_pt)
    else:
        print(f"❌ 错误: 找不到 images 目录 -> {img_dir}")
        print("请检查 --data_path 是否正确。")