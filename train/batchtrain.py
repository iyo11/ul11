import re
import sys
import warnings
import platform
import traceback
from pathlib import Path
from datetime import datetime

base_path = Path(__file__).resolve().parent.parent
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

import yaml
import torch
from ultralytics import YOLO
from utils.mail.email_sender import send_text_email, load_config as load_email_config

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def run_single_training(model_name_item, train_cfg, os_cfg, save_path, receiver):
    # --- 1. 配置提取 ---
    datasets_path = os_cfg["datasets_path"]
    batch_size = os_cfg["batch_size"]
    workers = os_cfg["workers"]
    cacheTF = os_cfg["cache"]
    epoch_count = train_cfg["epochs"]
    close_mosaic_count = train_cfg["close_mosaic"]
    datasets = train_cfg["dataset_yaml"]
    seed = train_cfg["seed"]
    optimizer = train_cfg["optimizer"]
    amp = train_cfg["amp"]
    patience = train_cfg["patience"]
    pretrained = train_cfg["pretrained"]
    module_edition = train_cfg["module_edition"]

    # --- 2. 解析 Run Name ---
    m = re.search(r"^yolo(\d+)([A-Za-z0-9]+)(?:_(.+))?\.yaml$", model_name_item)
    if not m:
        print(f"Skipping unsupported format: {model_name_item}")
        return
    version, variant, module = m.group(1), m.group(2), (m.group(3) or "base")
    dataset_name = Path(datasets).stem
    suffix = f"_{module_edition}" if module_edition != "e0" else ""
    run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}{suffix}_{dataset_name}_{epoch_count}_{seed}"

    # 发送 [STARTING] 消息
    _send_email_safe(receiver, f"[STARTING] {model_name_item}",
                     f"Training started for: {model_name_item}\nRun Name: {run_name}")

    # --- 3. 日志拦截器配置 ---
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    _stdout_orig, _stderr_orig = sys.stdout, sys.stderr
    _buf = []

    def _capture_write_factory(_orig_write):
        def _w(s):
            _orig_write(s)
            t = ansi_re.sub("", s)
            # 过滤掉频繁刷新的进度条，只保留关键节点
            if "100%" in t or not (("\r" in t) and ("it/s" in t or "%" in t)):
                _buf.append(t.replace("\r", ""))

        return _w

    sys.stdout.write = _capture_write_factory(_stdout_orig.write)
    sys.stderr.write = _capture_write_factory(_stderr_orig.write)

    status, err_text = "UNKNOWN", ""
    t0 = datetime.now()
    # 预设路径，防止训练未开始就崩溃导致变量未定义
    current_run_dir = save_path / run_name

    try:
        # --- 4. 执行训练 ---
        model_cfg = base_path / "models" / version / model_name_item
        model = YOLO(str(model_cfg))
        results = model.train(
            data=datasets_path + datasets, cache=cacheTF, imgsz=640,
            epochs=epoch_count, batch=batch_size, pretrained=pretrained,
            close_mosaic=close_mosaic_count, workers=workers, device="0",
            optimizer=optimizer, amp=amp, patience=patience,
            project=str(save_path), name=run_name, seed=seed, exist_ok=True,
        )
        current_run_dir = Path(results.save_dir)
        status = "FINISHED"

    except KeyboardInterrupt:
        # 用户手动中断，恢复终端后继续向上抛出
        sys.stdout.write, sys.stderr.write = _stdout_orig.write, _stderr_orig.write
        print("\n[INFO] Training interrupted by user.")
        raise KeyboardInterrupt

    except Exception:
        # 训练过程报错：记录报错堆栈
        status = "FAILED"
        err_text = traceback.format_exc()
        print(f"\n[ERROR] Task {model_name_item} failed.")

    finally:
        # --- 5. 善后处理与邮件发送 ---
        # 恢复终端标准输出
        sys.stdout.write, sys.stderr.write = _stdout_orig.write, _stderr_orig.write

        if status != "UNKNOWN":
            elapsed = datetime.now() - t0

            # 尝试保存内存中的日志到文件
            try:
                current_run_dir.mkdir(parents=True, exist_ok=True)
                with open(current_run_dir / "run.log", "w", encoding="utf-8", errors="ignore") as f:
                    f.writelines(_buf)
            except:
                pass

            # 构造邮件内容：直接从内存 _buf 取最后 50 行，确保即使没文件也能发日志
            last_console_logs = "".join(_buf[-50:]) if _buf else "No logs captured."

            subject = f"[{status}] {run_name}"
            email_body = (
                f"Status: {status}\n"
                f"Model: {model_name_item}\n"
                f"Elapsed: {str(elapsed)}\n\n"
            )

            if status == "FAILED":
                email_body += f"--- PYTHON TRACEBACK ---\n{err_text}\n\n"

            email_body += f"--- LAST CONSOLE OUTPUT ---\n{last_console_logs}"

            _send_email_safe(receiver, subject, email_body)


if __name__ == "__main__":
    try:
        full_cfg = load_train_config()
        train_cfg = full_cfg["train"]
        os_cfg = train_cfg["windows"] if platform.system() == "Windows" else train_cfg["linux"]
        save_path = base_path / "runs"
        receiver = load_email_config()["email"]["receiver"]

        model_names = train_cfg["model_name"]
        if isinstance(model_names, str):
            model_names = [model_names]

        for m_name in model_names:
            try:
                print(f"\n{'=' * 50}\n>>> CURRENT TASK: {m_name}\n{'=' * 50}")
                run_single_training(m_name, train_cfg, os_cfg, save_path, receiver)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(f"[CRITICAL ERROR] Failed to process {m_name}: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[STOP] Batch training cancelled by user.")
        sys.exit(0)