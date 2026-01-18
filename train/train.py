import warnings
from pathlib import Path

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ultralytics import YOLO

base_path = Path(__file__).resolve().parent.parent
save_path = base_path / "runs"
# 检测运行环境 Win or Linux
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

epoch_count = 300
close_mosaic_count = 45


if __name__ == '__main__':
    #model = YOLO('yolov8n.yaml')
    model = YOLO('../models/11/yolo11n.yaml')
    model.train(data= datasets_path +  '/NWPU_VHR.yaml',
                cache=cacheTF,
                imgsz=640,
                epochs=epoch_count,
                single_cls=False,
                batch=batch_size,
                close_mosaic=close_mosaic_count,
                mosaic=1.0,
                workers=workers,
                device='0',
                optimizer='SGD', # using SGD
                resume=False,
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                patience=0,
                project=str(save_path),
                name='11n_NWPU_300',
                #save_period=20,
                #固定随机种子
                seed=0
                )
