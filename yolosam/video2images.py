import cv2
import os
import argparse
import sys

def video_to_frames(video_path, output_dir, skip_interval=1):
    # 1. 检查视频
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频: {video_path}")
        sys.exit(1)

    # 2. 准备 input 目录 (注意：这里直接存到 input，方便后续 COLMAP 读取)
    images_dir = os.path.join(output_dir, "input")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"📂 处理视频: {video_path}")
    print(f"📂 输出图片到: {images_dir}")

    cap = cv2.VideoCapture(video_path)
    count = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % skip_interval == 0:
            file_name = f"{save_count:05d}.jpg"
            save_path = os.path.join(images_dir, file_name)
            cv2.imwrite(save_path, frame)
            save_count += 1
        count += 1

    cap.release()
    print(f"✅ 拆帧完成！共 {save_count} 张图片。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--skip", type=int, default=1)
    args = parser.parse_args()

    video_to_frames(args.video, args.outdir, args.skip)