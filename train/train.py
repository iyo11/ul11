import re
import sys
import warnings
from pathlib import Path
import platform

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ultralytics import YOLO

if __name__ == '__main__':
    base_path = Path(__file__).resolve().parent.parent
    save_path = base_path / "runs"
    if platform.system() == 'Windows':
        datasets_path = '../datasets_local'
        batch_size = 8
        workers = 4
        cacheTF = False
    else:
        datasets_path = '../datasets'
        batch_size = 24
        workers = 10
        cacheTF = True

    epoch_count = 300
    close_mosaic_count = 45
    model_name = "yolo11n_SADConv.yaml"
    datasets = '/NWPU_VHR.yaml'
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

    dataset_name = Path(datasets).stem
    if module_edition != "e0":
        run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}_{module_edition}_{dataset_name}_{epoch_count}_{seed}"
    else:
        run_name = f"{dataset_name}/v{version}_seed_{seed}/{version}{variant}_{module}_{dataset_name}_{epoch_count}_{seed}"
    run_dir = save_path / run_name
    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    _stdout_orig = sys.stdout
    _stderr_orig = sys.stderr
    _stdout_write_orig = sys.stdout.write
    _stderr_write_orig = sys.stderr.write
    _buf = []
    sys.stdout.write = lambda s, _ow=_stdout_write_orig, _b=_buf, _ar=ansi_re: (
        _ow(s),
        (
            (lambda t:
             _b.append(t.replace("\r", "")) if ("100%" in t)
             else (None if ("\r" in t and ("it/s" in t or "%" in t)) else _b.append(t))
             )(_ar.sub("", s))
        )
    )[0]
    sys.stderr.write = lambda s, _ow=_stderr_write_orig, _b=_buf, _ar=ansi_re: (
        _ow(s),
        (
            (lambda t:
             _b.append(t.replace("\r", "")) if ("100%" in t)
             else (None if ("\r" in t and ("it/s" in t or "%" in t)) else _b.append(t))
             )(_ar.sub("", s))
        )
    )[0]
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
        device='0',
        optimizer=optimizer,
        resume=False,
        amp=amp,
        patience=patience,
        project=str(save_path),
        name=run_name,
        seed=seed
    )
    save_dir = Path(getattr(results, "save_dir", run_dir))
    sys.stdout.write = _stdout_write_orig
    sys.stderr.write = _stderr_write_orig
    sys.stdout = _stdout_orig
    sys.stderr = _stderr_orig
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "run.log"
    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        f.writelines(_buf)
    print(f"[OK] run.log saved to: {log_path}")

