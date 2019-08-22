#coding:utf-8
#author: lxz-hxy

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
from cv2 import cv2 as cv2
import argparse
import time
from PIL import Image 
from timeit import default_timer as timer

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

def detect_imgs(testFolder,detectors):
    print("Start testing in %s"%(testFolder))
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    testImages = []
    for name in os.listdir(testFolder):
        testImages.append(os.path.join(testFolder, name))
    testDatas = TestLoader(testImages)
    print('Now to detect.....')
    allBoxes, _ = mtcnnDetector.detect_face(testDatas)
    print('\n')

    savePath = os.path.join(rootPath, 'testing', 'results')
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    # Save it
    for idx, imagePath in enumerate(testImages):
        image = cv2.imread(imagePath)
        count = 0
        for bbox in allBoxes[idx]: 
            #cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255)) #识别为人脸的概率值
            '''可以适当调节一下画框的范围,防止人脸部位漏剪'''
            x1 = int(bbox[0]) 
            y1 = int(bbox[1]) 
            x2 = int(bbox[2]) 
            y2 = int(bbox[3]) 
            #cv2.rectangle(image, (x1,y1),(x2,y2),(0,0,255))
            image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (96,96))
            cv2.imwrite(os.path.join(savePath, "%d_%d.jpg" %(count,idx)), image)
            count += 1
            image = cv2.imread(imagePath)
            print('left:({},{}),right:({},{})'.format(x1,y1,x2,y2))
        print('Pic detected {} faces!!!\n'.format(count))

'''
//代码需要优化, 采用多线程
读取摄像头的视频流，保存图片帧，从而进行人脸检测抠图 
'''
def detect_video(testFolder, detectors):
    print('Start testing in video.....\n')
    savePath = os.path.join(rootPath, 'testing', 'results')
    if not os.path.isdir(savePath):
        os.mkdir(savePath)

    frame_interval = 10
    frame_count = 0
    testImages = []
    url = 'rtsp://admin:px68018888@192.168.63.189:554' #公司摄像头
    #video = './video/1.mp4' 
    cap = cv2.VideoCapture(url)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    ret, frame = cap.read()
    while ret:
        frame_count += 1
        ret, frame = cap.read()
        if (frame_count % frame_interval) == 0:
            pic_path = testFolder + '/' + '%d.jpg'%(frame_count)
            #cv2.imshow('ori', frame)
            #cv2.waitKey(200)
            cv2.imwrite(pic_path, frame)
            testImages.append(pic_path)
            testDatas = TestLoader(testImages)
            allBoxes, _ = mtcnnDetector.detect_face(testDatas)
            if len(allBoxes) == 0:
                continue
            else:
                for idx,imagePath in enumerate(testImages):
                    image = cv2.imread(imagePath)
                    tmp =[]
                    for bbox in allBoxes[idx]:
                        tmp.append(bbox)
                        for i in range(len(tmp)):
                            x1 = int(tmp[i][0]) 
                            y1 = int(tmp[i][1]) 
                            x2 = int(tmp[i][2]) 
                            y2 = int(tmp[i][3])
                            #cv2.rectangle(image, (x1, y1),(x2, y2), (0,0,255))
                            print('Face detected: left:({},{}),right:({},{})'.format(x1,y1,x2,y2))
                            image = image[y1:y2, x1:x2]
                            #cv2.imshow('crop_face',image)
                            #cv2.waitKey(100)
                            cv2.imwrite(os.path.join(savePath, "%d_%d.jpg" %(frame_count,i)), image)
                            testImages = []
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 


'''
realtime检测
'''
def detect_url(detectors):
    print('Start testing in video.....\n')
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    '''
    rtsp://admin:px68018888@192.168.63.189:554 #公司摄像头
    rtsp://admin:12345daoge@172.16.6.3:554/Streaming/Channels/1301 #小区门口监控
    rtsp://admin:px68018888@172.16.12.79:554 #万达步行街
    rtsp://admin:liguowei123456@172.16.12.65:554 #万达A门
    rtsp://admin:liguowei123456@172.16.12.75:554 #万达C门地下室电梯出口

    '''

    url = 'rtsp://admin:px68018888@172.16.12.79:554'
    #video = 'video_test.avi'
    cap = cv2.VideoCapture(url)
    cap.set(3, 340)
    cap.set(4, 480)
    corpbbox = None
    while True:
        #fps = cap.get(cv2.CAP_PROP_FPS)
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if ret:
            image = cv2.resize(frame, (frame.shape[1]/3, frame.shape[0]/3))
            image = np.array(image)
            boxes_c,landmarks = mtcnnDetector.detect_video(image)
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t
            print(fps)
            print(boxes_c)
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                #if score > thresh:
                cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                            (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                cv2.putText(image, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
            cv2.putText(image, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i])/2):
                    cv2.circle(image, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))            
            # # time end
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
    detectors=net(stage)
    detect_imgs('/home/lxz/hxy/mtcnn/testing/images/', detectors)
    # detect_video('/home/lxz/hxy/project/mtcnn/testing/images/', detectors) #检测视频流
    #detect_url(detectors) #realtime检测
