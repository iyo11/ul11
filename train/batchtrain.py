import re
import sys
import warnings
import platform
import traceback
from pathlib import Path
from datetime import datetime

import torch
import yaml
from ultralytics import YOLO

# 路径初始化
base_path = Path(__file__).resolve().parent.parent
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

from utils.mail.email_sender import send_text_email, load_config as load_email_config

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- 辅助函数 ---
def _read_text_safe(p: Path, limit: int = 150_000) -> str:
    """安全读取日志文件"""
    if not p.exists(): return ""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) <= limit: return text
        return text[:80_000] + "\n\n--- LOG TRUNCATED ---\n\n" + text[-60_000:]
    except:
        return ""


def _send_email_safe(receiver: str, subject: str, content: str) -> None:
    """安全发送邮件，失败不阻断流程"""
    try:
        send_text_email(receiver, subject, content)
        print(f">> Email notification sent: {subject}")
    except Exception as e:
        print(f"!! Email notification failed: {e}")


def load_train_config():
    config_path = base_path / "config/batchtrain.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- 日志拦截工厂 ---
ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def get_capture_writer(original_writer, buffer_list):
    def _w(s):
        original_writer(s)  # 仍然输出到控制台，不影响此时的查看
        t = ansi_re.sub("", s)
        # 过滤掉进度条刷新，避免日志过大，只保留关键信息和换行
        if "100%" in t:
            buffer_list.append(t.replace("\r", ""))
        elif not (("\r" in t) and ("it/s" in t or "%" in t)):
            buffer_list.append(t.replace("\r", ""))

    return _w


if __name__ == "__main__":
    save_path = base_path / "runs"

    # Global Try-Catch for Config Loading
    try:
        # 1. 加载配置
        full_cfg = load_train_config()
        train_cfg = full_cfg["train"]

        # 平台特定配置
        os_cfg = train_cfg["windows"] if platform.system() == "Windows" else train_cfg["linux"]

        # 统一处理路径
        d_path_str = os_cfg["datasets_path"].rstrip("/\\")
        d_yaml_str = train_cfg["dataset_yaml"].lstrip("/\\")
        dataset_full_path = str(Path(d_path_str) / d_yaml_str)

        # 获取模型列表
        model_names = train_cfg["model_name"]
        if isinstance(model_names, str):
            model_names = [model_names]

        # 邮件配置
        email_cfg = load_email_config()["email"]
        receiver = email_cfg["receiver"]

    except Exception as e:
        traceback.print_exc()
        sys.exit("Config loading failed.")

    # 保存原始输出流
    _stdout_orig, _stderr_orig = sys.stdout, sys.stderr

    print(f"Total models to train: {len(model_names)}")

    # 2. 循环遍历模型列表进行训练
    for i, model_file in enumerate(model_names):

        # 每个模型重置日志 buffer
        _buf = []

        # 挂载日志拦截 (每个循环重新挂载，确保状态清晰)
        sys.stdout.write = get_capture_writer(_stdout_orig.write, _buf)
        sys.stderr.write = get_capture_writer(_stderr_orig.write, _buf)

        t0 = datetime.now()
        run_name = "unknown"
        final_save_dir = Path("unknown")

        try:
            print(f"\n{'=' * 20} Processing Model {i + 1}/{len(model_names)}: {model_file} {'=' * 20}\n")

            # --- 解析配置 ---
            m = re.search(r"^yolo(\d+)([A-Za-z0-9]+)(?:_(.+))?\.yaml$", model_file)
            if not m:
                raise ValueError(f"Unsupported model_name format: {model_file}")

            version, variant, module = m.group(1), m.group(2), (m.group(3) or "base")

            dataset_name = Path(train_cfg["dataset_yaml"]).stem
            module_edition = train_cfg["module_edition"]
            suffix = f"_{module_edition}" if module_edition != "e0" else ""
            seed = train_cfg["seed"]
            epoch_count = train_cfg["epochs"]

            # 构建 run_name
            run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}{suffix}_{dataset_name}_{epoch_count}_{seed}"

            # 模型路径
            model_cfg_path = base_path / "models" / version / model_file

            # ==========================================
            # [新增功能] 训练开始前发送邮件
            # ==========================================
            start_subject = f"[STARTING] {model_file} ({i + 1}/{len(model_names)})"
            start_content = (
                f"Training Started at: {t0.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Model: {model_file}\n"
                f"Run Name: {run_name}\n"
                f"Dataset: {dataset_full_path}\n"
                f"Epochs: {epoch_count}\n"
                f"Batch Size: {os_cfg['batch_size']}\n"
                f"Device: {os_cfg.get('device', '0')}\n"
            )
            # 使用原始 stdout 打印，避免被拦截逻辑混淆（虽然拦截器也会转发）
            _send_email_safe(receiver, start_subject, start_content)

            # --- 初始化与训练 ---
            model = YOLO(str(model_cfg_path))

            results = model.train(
                data=dataset_full_path,
                cache=os_cfg["cache"],
                imgsz=640,
                epochs=epoch_count,
                batch=os_cfg["batch_size"],
                pretrained=train_cfg["pretrained"],
                close_mosaic=train_cfg["close_mosaic"],
                workers=os_cfg["workers"],
                device="0",
                optimizer=train_cfg["optimizer"],
                amp=train_cfg["amp"],
                patience=train_cfg["patience"],
                project=str(save_path),
                name=run_name,
                seed=seed,
                exist_ok=True,
            )

            # --- 训练后处理 ---
            # 1. 保存日志
            final_save_dir = Path(results.save_dir)
            log_path = final_save_dir / "run.log"
            with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
                f.writelines(_buf)

            # 2. 发送成功邮件
            elapsed = datetime.now() - t0
            log_content = _read_text_safe(log_path)
            finish_subject = f"[FINISHED] {run_name}"
            finish_content = (
                f"Status: SUCCESS\n"
                f"Model: {model_file}\n"
                f"Save Dir: {final_save_dir}\n"
                f"Elapsed: {str(elapsed)}\n\n"
                f"---- Last Logs ----\n"
                f"{log_content}"
            )
            _send_email_safe(receiver, finish_subject, finish_content)

        except Exception as e:
            # 发生错误时，先恢复标准流以便在控制台打印 Traceback
            sys.stdout.write = _stdout_orig.write
            sys.stderr.write = _stderr_orig.write

            error_msg = traceback.format_exc()
            print(error_msg)

            # 发送报错邮件
            fail_subject = f"[FAILED] {model_file} - {run_name}"
            fail_content = (
                f"Error during training {model_file}:\n\n"
                f"{error_msg}\n\n"
                f"---- Captured Logs (Last 50 lines) ----\n"
                f"{''.join(_buf[-50:])}"
            )
            _send_email_safe(receiver, fail_subject, fail_content)

        finally:
            # 必须恢复标准输出流，否则下一次循环或程序退出时会出错
            sys.stdout.write = _stdout_orig.write
            sys.stderr.write = _stderr_orig.write

            # 强制清理内存
            import gc

            if 'model' in locals(): del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()