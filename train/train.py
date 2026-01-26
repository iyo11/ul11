import re
import sys
import warnings
from pathlib import Path
import platform
import traceback
from datetime import datetime

import yaml

from utils.mail.email_sender import send_text_email, load_config as load_email_config

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ultralytics import YOLO


def _read_text_safe(p: Path, limit: int = 150_000) -> str:
    """Read text file safely and truncate if too large."""
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= limit:
        return text
    head = text[:80_000]
    tail = text[-60_000:]
    return head + "\n\n--- LOG TRUNCATED (middle omitted) ---\n\n" + tail


def _send_email_safe(receiver: str, subject: str, content: str) -> None:
    """Send email but never raise."""
    try:
        send_text_email(receiver, subject, content)
    except Exception as e:
        print(f"Email notification failed: {e}")

def load_train_config():
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config/train.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent
    save_path = base_path / "runs"
    train_cfg = load_train_config()["train"]
    # Platform-specific config
    if platform.system() == "Windows":
        os_cfg = train_cfg["windows"]
    else:
        os_cfg = train_cfg["linux"]

    datasets_path = os_cfg["datasets_path"]
    batch_size = os_cfg["batch_size"]
    workers = os_cfg["workers"]
    cacheTF = os_cfg["cache"]
    # Common training config
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
    # email receiver
    email_cfg = load_email_config()["email"]
    receiver = email_cfg["receiver"]
    # -------------------------------

    # Parse model name
    m = re.search(r"^yolo(\d+)([A-Za-z0-9]+)(?:_(.+))?\.yaml$", model_name)
    if not m:
        raise ValueError(
            f"Unsupported model_name format: {model_name}. Example: yolo11n.yaml or yolo11n_xxx.yaml"
        )
    version = m.group(1)
    variant = m.group(2)
    module = m.group(3) or "base"

    dataset_name = Path(datasets).stem
    if module_edition != "e0":
        run_name = (
            f"{dataset_name}/v{version}_seed_{seed}/"
            f"{version}{variant}_{module}_{module_edition}_{dataset_name}_{epoch_count}_{seed}"
        )
    else:
        run_name = (
            f"{dataset_name}/v{version}_seed_{seed}/"
            f"{version}{variant}_{module}_{dataset_name}_{epoch_count}_{seed}"
        )

    run_dir = save_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture logs (stdout/stderr)
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    _stdout_orig = sys.stdout
    _stderr_orig = sys.stderr
    _stdout_write_orig = sys.stdout.write
    _stderr_write_orig = sys.stderr.write
    _buf = []

    def _capture_write_factory(_orig_write):
        def _w(s):
            _orig_write(s)
            t = ansi_re.sub("", s)
            # progress bars contain \r updates; skip noisy interim lines
            if "100%" in t:
                _buf.append(t.replace("\r", ""))
            else:
                if ("\r" in t) and ("it/s" in t or "%" in t):
                    return
                _buf.append(t.replace("\r", ""))
        return _w

    sys.stdout.write = _capture_write_factory(_stdout_write_orig)
    sys.stderr.write = _capture_write_factory(_stderr_write_orig)

    results = None
    save_dir = run_dir
    log_path = run_dir / "run.log"
    status = "UNKNOWN"
    err_text = ""
    t0 = datetime.now()

    try:
        model_cfg = Path("..") / "models" / version / model_name
        model = YOLO(str(model_cfg))

        results = model.train(
            data=datasets_path + datasets,
            cache=cacheTF,
            imgsz=640,
            epochs=epoch_count,
            single_cls=False,
            batch=batch_size,
            pretrained=pretrained,
            close_mosaic=close_mosaic_count,
            mosaic=1.0,
            workers=workers,
            device="0",
            optimizer=optimizer,
            resume=False,
            amp=amp,
            patience=patience,
            project=str(save_path),
            name=run_name,
            seed=seed,
        )

        # Ultralytics usually sets results.save_dir
        save_dir = Path(getattr(results, "save_dir", run_dir))
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "run.log"
        status = "FINISHED"

    except Exception:
        status = "FAILED"
        err_text = traceback.format_exc()

        # If save_dir isn't available, still keep log under run_dir
        save_dir = run_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "run.log"

    finally:
        # Restore stdout/stderr no matter what
        sys.stdout.write = _stdout_write_orig
        sys.stderr.write = _stderr_write_orig
        sys.stdout = _stdout_orig
        sys.stderr = _stderr_orig

        # Write run.log
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
                f.writelines(_buf)
            print(f"[OK] run.log saved to: {log_path}")
        except Exception as e:
            print(f"Failed to write run.log: {e}")

        # Email notification (never raise)
        try:
            elapsed = datetime.now() - t0
            log_text = _read_text_safe(log_path, limit=150_000)

            subject = f"[{status}] {run_name}"
            header = (
                f"Status: {status}\n"
                f"Run name: {run_name}\n"
                f"Dataset: {datasets_path + datasets}\n"
                f"Model: {model_name}\n"
                f"Epochs: {epoch_count}\n"
                f"Seed: {seed}\n"
                f"Optimizer: {optimizer}\n"
                f"AMP: {amp}\n"
                f"Pretrained: {pretrained}\n"
                f"Save dir: {save_dir}\n"
                f"Log path: {log_path}\n"
                f"Elapsed: {str(elapsed)}\n\n"
            )

            if status == "FAILED":
                content = (
                    header
                    + "---- Exception Traceback ----\n"
                    + (err_text or "No traceback captured.\n")
                    + "\n---- Captured Log (run.log) ----\n"
                    + (log_text or "(empty)\n")
                )
            else:
                content = (
                    header
                    + "Training finished successfully.\n\n"
                    + "---- Captured Log (run.log) ----\n"
                    + (log_text or "(empty)\n")
                )

            _send_email_safe(receiver, subject, content)

        except Exception as e:
            print(f"Email notification build/send failed: {e}")

    # If failed, keep non-zero exit code (optional but useful for scripts)
    if status == "FAILED":
        sys.exit(1)
