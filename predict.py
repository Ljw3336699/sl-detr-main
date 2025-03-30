import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':
    model = RTDETR('') # select your model.pt path
    # print(model.info(detailed=True))
    model.predict(source='',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )
