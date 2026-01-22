import re
import sys
import warnings
import platform
from pathlib import Path
from ultralytics import YOLO

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

base_path = Path(__file__).resolve().parent.parent
save_path = base_path / "runs"

if platform.system() == 'Windows':
    datasets_root = Path('../datasets_local')
    batch_size = 8
    workers = 4
    cacheTF = False
else:
    datasets_root = Path('../datasets')
    batch_size = 24
    workers = 10
    cacheTF = True

epoch_count = 300
close_mosaic_count = 45
model_name = "yolo11n_GAM.yaml"
dataset_yaml = '/NWPU_VHR.yaml'
seed = 42
optimizer = 'SGD'
amp = False
patience = 0
pretrained = True
module_edition = "e0"

m = re.search(r"^yolo(\d+)([A-Za-z0-9]+)(?:_(.+))?\.yaml$", model_name)
if not m:
    raise ValueError(f"model_name 格式不支持: {model_name}. 例: yolo11n.yaml 或 yolo11n_xxx.yaml")
version = m.group(1)
variant = m.group(2)
module = m.group(3) or "base"
dataset_name = Path(dataset_yaml).stem
if module_edition != "e0":
    run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}_{module_edition}_{dataset_name}_{epoch_count}_{seed}"
else:
    run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}_{dataset_name}_{epoch_count}_{seed}"

run_dir = save_path / run_name

class DualLogger:
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')
        self.ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # 用于去除控制台颜色代码
    def write(self, message):
        # 1. 打印到屏幕
        self.terminal.write(message)
        self.terminal.flush()
        # 2. 写入到文件 (去除颜色代码)
        clean_msg = self.ansi_re.sub("", message)
        # 过滤逻辑：忽略刷屏的进度条（带 \r），只记录最终完成的（带 100%）
        if "\r" not in clean_msg or "100%" in clean_msg:
            self.log.write(clean_msg.replace("\r", ""))
            self.log.flush()  # 【关键】强制写入硬盘

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

if __name__ == '__main__':
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    print(f"LOG SAVE PATH: {log_path}")
    sys.stdout = DualLogger(log_path, sys.stdout)
    sys.stderr = DualLogger(log_path, sys.stderr)

    try:
        model_cfg = Path("..") / "models" / version / model_name
        data_path = str(datasets_root) + dataset_yaml
        model = YOLO(str(model_cfg))
        results = model.train(
            data=data_path,
            cache=cacheTF,
            imgsz=640,
            epochs=epoch_count,
            single_cls=False,
            batch=batch_size,
            pretrained=pretrained,
            close_mosaic=close_mosaic_count,
            mosaic=1.0,
            workers=workers,
            device='0',
            optimizer=optimizer,
            resume=False,
            amp=amp,
            patience=patience,
            project=str(save_path),
            name=run_name,
            seed=seed,
            exist_ok=True
        )
        print(f"LOG SAVE PATH: {log_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise e

    finally:
        # 恢复标准输出流
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.stdout.terminal
        sys.stderr = sys.stderr.terminal