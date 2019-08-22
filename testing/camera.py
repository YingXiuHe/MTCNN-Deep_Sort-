#coding:utf-8
#author: lxz-hxy

'''
MTCNN在图片以及视频流上进行人脸检测，
并且将检测到的人脸从图像或者视频流中截取下来保存到指定目录中；
'''
import tensorflow as tf
import numpy as np
import os
import sys
import cv2 as cv2

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector

import time 
import gc
from multiprocessing import Process, Manager


def net(stage):
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/pnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net,modelPath) 
    if stage in ['rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/rnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/onet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath)
    return detectors


'''
python多线程：
receive：该线程接收图片
realse：该线程处理图片，进行人脸检测
'''
def receive(stack):

    top = 100
    # url = ''
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        if ret:
            stack.append(frame)
            if len(stack) >= top:
                del stack[:50]
                gc.collect()

 
def realse(stack):
    print('Begin to get frame......')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    detectors = net('onet')
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    while True:
        
        if len(stack) > 30:
            image = stack.pop()
            image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])))
            image = np.array(image)
            boxes_c, _ = mtcnnDetector.detect_video(image)            
            for bbox in boxes_c:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,0.3,color=(0,255,0))
                cv2.rectangle(image, (x1, y1), (x2,y2), (0,0,255))
                
                # cut = image[y1:y2, x1:x2]
                # for i in range(len(boxes_c)):
                #     cv2.imwrite(str(i) + '.jpg', cut)
                
                print('deteced face: ({},{}), ({},{})'.format(x1, y1, x2, y2))   
            cv2.imshow("Detected", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    t = Manager().list()
    t1 = Process(target=receive, args=(t,))
    t2 = Process(target=realse, args=(t,))
    t1.start()
    t2.start()
    t1.join()
    t2.terminate()
    
