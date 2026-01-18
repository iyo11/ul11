import re
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ultralytics import YOLO

base_path = Path(__file__).resolve().parent.parent
save_path = base_path / "runs"
import platform
if platform.system() == 'Windows':
    datasets_path = '../datasets_local'
    batch_size = 8
    workers = 4
    cacheTF =  False
else:
    datasets_path = '../datasets'
    batch_size = 24
    workers = 10
    cacheTF =  True


#config
epoch_count = 300
close_mosaic_count = 45
model_name = "yolo11n_Converse2D.yaml"
datasets = '/NWPU_VHR.yaml'
seed = 11
optimizer = 'SGD'
amp = False
patience=0
pretrained=True
module_edition="e1"
#config end



m = re.search(r"^yolo(\d+)([A-Za-z0-9]+)(?:_(.+))?\.yaml$", model_name)
if not m:
    raise ValueError(f"model_name 格式不支持: {model_name}. 例: yolo11n.yaml 或 yolo11n_xxx.yaml")
version = m.group(1)
variant = m.group(2)
module  = m.group(3) or "base"
dataset_name = Path(datasets).stem
if module_edition != "e0":
    run_name = f"{dataset_name}/{version}/seed_{seed}/{version}{variant}_{module}_{module_edition}_{dataset_name}_{epoch_count}_{seed}"
else:
    run_name = f"{dataset_name}/{version}/seed_{seed}/{version}{variant}_{module}_{dataset_name}_{epoch_count}_{seed}"
if __name__ == '__main__':
    model_cfg = Path("..") / "models" / version / model_name
    model = YOLO(str(model_cfg))
    model.train(data= datasets_path + datasets,
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
