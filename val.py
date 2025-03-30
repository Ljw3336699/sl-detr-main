import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':

    model = RTDETR('') # select your model.pt path
    model.val(data='',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=1,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
