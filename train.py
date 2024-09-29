import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s-p2_mul2_C2f_ODConv-SEAMHead.yaml')
    model.train(data=r"D:\PyCharm\DataSets\Noodle_DataSets\DataSets.yaml",
                cache=False,
                imgsz=200,
                epochs=100,
                batch=4,
                close_mosaic=0,
                amp=False, # close amp
                project='runs/train',
                name='yolov8s-p2_mul2_C2f_ODConv-SEAMHead'
                )