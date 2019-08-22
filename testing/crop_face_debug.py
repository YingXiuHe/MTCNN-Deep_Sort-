#coding:utf-8
#author: lxz-hxy

'''
MTCNN在图片以及视频流上进行人脸检测，
并且将检测到的人脸从图像或者视频流中扣下来保存到指定目录中；
'''

import tensorflow as tf
import numpy as np
import os
import sys
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from cv2 import cv2 
import argparse
import time
import queue
import gc
import threading
from multiprocessing import Process, Manager
from PIL import Image 


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



#检测图片
def detect_imgs(testFolder,detectors):
    print("Start testing in %s"%(testFolder))
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    testImages = [] #图片的绝对路径
    for name in os.listdir(testFolder):
        testImages.append(os.path.join(testFolder, name))

    testDatas = TestLoader(testImages)
    allBoxes, _ = mtcnnDetector.detect_face(testDatas)
    print('\n')
    
    savePath = os.path.join(rootPath, 'testing', 'results')
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    # Save it
    for idx, imagePath in enumerate(testImages):
        image = cv2.imread(imagePath)
        tmp = [] 
        for bbox in allBoxes[idx]:
            tmp.append(bbox)
            for i in range(len(tmp)):
                x1 = int(tmp[i][0]) 
                y1 = int(tmp[i][1]) 
                x2 = int(tmp[i][2]) 
                y2 = int(tmp[i][3])
                #cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255)) #识别为人脸的概率值
                #cv2.rectangle(image, (x1,y1),(x2,y2),(0,0,255))

                image = image[int(tmp[i][1]):int(tmp[i][3]),int(tmp[i][0]):int(tmp[i][2])]
                image = cv2.resize(image, (96,96))
                
                cv2.imwrite(os.path.join(savePath, "%d_%d.jpg" %(i,idx)), image)
                image = cv2.imread(imagePath)
                print('left:({},{}),right:({},{})'.format(x1,y1,x2,y2))



def detect_url(detectors):
    print('Start testing in video.....\n')
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    
    '''
    rtsp://admin:px68018888@192.168.63.189:554 #公司摄像头
    rtsp://admin:12345daoge@172.16.6.3:554/Streaming/Channels/1301 #小区门口监控
    rtsp://admin:px68018888@172.16.12.79:554 #万达步行街
    rtsp://admin:liguowei123456@172.16.12.65:554 #万达内部
    '''

    url = 'rtsp://admin:liguowei123456@172.16.12.65:554' 
    cap = cv2.VideoCapture(url)
    cap.set(3, 340)
    cap.set(4, 480)
    while True:
        #fps = cap.get(cv2.CAP_PROP_FPS)
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if ret:
            image = cv2.resize(frame, (frame.shape[1]/3, frame.shape[0]/3))
            image = np.array(image)
            t2 = cv2.getTickCount()
            print('Get frame cost time is:{} ms'.format((t2-t1)/cv2.getTickFrequency()*1000))
            boxes_c, _ = mtcnnDetector.detect_video(image)
            t3 = cv2.getTickCount()
            t = (t3 - t2) / cv2.getTickFrequency()
            print('Detect spend time is: {} ms'.format(t*1000))
            print('\n')
            fps = 1.0 / t
            for bbox in boxes_c:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cv2.rectangle(image, (x1, y1), (x2,y2), (0,0,255))
                print('deteced face: ({},{}), ({},{})'.format(x1, y1, x2, y2))   
            ## time end
            cv2.putText(image, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)
            cv2.imshow("Detect", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('No video input detected.......')
            break
    cap.release()
    cv2.destroyAllWindows()
     
if __name__ == "__main__":
    stage = 'onet' #选择网络
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 设定GPU 可多块 '0,1,2'
    
    detectors = net(stage)
    '''
    可设置arg参数选择检测模式
    '''
    #detect_imgs('/home/lxz/hxy/project/mtcnn/testing/images/', detectors) #检测照片
    detect_url(detectors) #检测视频流
    
