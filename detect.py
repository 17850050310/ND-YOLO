import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/DataSets_2093_514_clean_2_limitArea0.000025/yolov8s/weights/best.pt') # select your model.pt path
    model.predict(source=r"D:\PyCharm\DataSets\Noodle_DataSets\DataSets_2093_514_clean_2_limitArea0.000025\valid\images",
                  imgsz=1280,
                  project='runs/val/Contrast_2093_514_clean_2_limitArea0.000025',
                  name='yolov8s',
                  # save=True,
                  conf=0.4,
                  # visualize=True # visualize model features maps
                  save_txt=True,
                  save_conf=True,
                  show_conf=True
                )