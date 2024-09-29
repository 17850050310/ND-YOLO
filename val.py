import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\PyCharm\YOLO\YOLOv8\ultralytics-main\runs\train\DataSets_2093_514_clean_2_limitArea0.000025\up_yolov8-p2_mul2_C2f_ODConv-V2-SEAMHead\weights\best.pt")
    model.val(data=r"D:\PyCharm\DataSets\Noodle_DataSets\DataSets.yaml",
              split='val',
              imgsz=1280,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/Contrast_2093_514_clean_2_limitArea0.000025',
              name='yolov8-p2_mul2_C2f_ODConv-V2-SEAMHead-train',
              plots=True,
              # save_hybrid=True
              )