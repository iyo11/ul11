import re
import sys
import warnings
import platform
import traceback
from pathlib import Path

from datetime import datetime

# 路径初始化
base_path = Path(__file__).resolve().parent.parent
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

import yaml
import torch
from ultralytics import YOLO
from utils.mail.email_sender import send_text_email, load_config as load_email_config

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _read_text_safe(p: Path, limit: int = 150_000) -> str:
    """安全读取文本，防止 log 过大导致内存溢出"""
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) <= limit:
            return text
        return text[:80_000] + "\n\n--- LOG TRUNCATED ---\n\n" + text[-60_000:]
    except:
        return ""


def _send_email_safe(receiver: str, subject: str, content: str) -> None:
    try:
        send_text_email(receiver, subject, content)
    except Exception as e:
        print(f"Email notification failed: {e}")


def load_train_config():
    config_path = base_path / "config/train.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    save_path = base_path / "runs"
    full_cfg = load_train_config()
    train_cfg = full_cfg["train"]

    # 平台特定配置
    os_cfg = train_cfg["windows"] if platform.system() == "Windows" else train_cfg["linux"]

    datasets_path = os_cfg["datasets_path"]
    batch_size = os_cfg["batch_size"]
    workers = os_cfg["workers"]
    cacheTF = os_cfg["cache"]

    # 训练通用配置
    epoch_count = train_cfg["epochs"]
    close_mosaic_count = train_cfg["close_mosaic"]
    model_name = train_cfg["model_name"]
    datasets = train_cfg["dataset_yaml"]
    seed = train_cfg["seed"]
    optimizer = train_cfg["optimizer"]
    amp = train_cfg["amp"]
    patience = train_cfg["patience"]
    pretrained = train_cfg["pretrained"]
    module_edition = train_cfg["module_edition"]

    # 邮件接收者
    email_cfg = load_email_config()["email"]
    receiver = email_cfg["receiver"]

    # 解析模型名称生成 run_name
    m = re.search(r"^yolo(\d+)([A-Za-z0-9]+)(?:_(.+))?\.yaml$", model_name)
    if not m:
        raise ValueError(f"Unsupported model_name format: {model_name}")

    version, variant, module = m.group(1), m.group(2), (m.group(3) or "base")
    dataset_name = Path(datasets).stem

    suffix = f"_{module_edition}" if module_edition != "e0" else ""
    run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}{suffix}_{dataset_name}_{epoch_count}_{seed}"

    # 日志拦截逻辑
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    _stdout_orig, _stderr_orig = sys.stdout, sys.stderr
    _buf = []


    def _capture_write_factory(_orig_write):
        def _w(s):
            _orig_write(s)
            t = ansi_re.sub("", s)
            if "100%" in t:
                _buf.append(t.replace("\r", ""))
            elif not (("\r" in t) and ("it/s" in t or "%" in t)):
                _buf.append(t.replace("\r", ""))

        return _w


    sys.stdout.write = _capture_write_factory(_stdout_orig.write)
    sys.stderr.write = _capture_write_factory(_stderr_orig.write)

    # --- 训练核心块 ---
    t0 = datetime.now()

    try:
        model_cfg = base_path / "models" / version / model_name
        model = YOLO(str(model_cfg))

        results = model.train(
            data=datasets_path + datasets,
            cache=cacheTF,
            imgsz=640,
            epochs=epoch_count,
            batch=batch_size,
            pretrained=pretrained,
            close_mosaic=close_mosaic_count,
            workers=workers,
            device="0",
            optimizer=optimizer,
            amp=amp,
            patience=patience,
            project=str(save_path),
            name=run_name,
            seed=seed,
            exist_ok=True,
        )

        # 1. 只有成功运行到这里，才会执行后续逻辑
        # 先恢复输出流，确保后续打印正常
        sys.stdout.write, sys.stderr.write = _stdout_orig.write, _stderr_orig.write

        # 2. 保存日志文件
        final_save_dir = Path(results.save_dir)
        log_path = final_save_dir / "run.log"
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.writelines(_buf)

        # 3. 发送邮件
        elapsed = datetime.now() - t0
        log_content = _read_text_safe(log_path)
        subject = f"[FINISHED] {run_name}"
        header = (
            f"Status: FINISHED\n"
            f"Model: {model_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Save Dir: {final_save_dir}\n"
            f"Elapsed: {str(elapsed)}\n\n"
        )
        email_body = header + "Success!\n\n---- Last Logs ----\n" + log_content
        _send_email_safe(receiver, subject, email_body)

    except Exception:
        sys.stdout.write, sys.stderr.write = _stdout_orig.write, _stderr_orig.write
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout.write, sys.stderr.write = _stdout_orig.write, _stderr_orig.write