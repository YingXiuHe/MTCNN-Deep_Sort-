#-*-coding: utf-8-*-
#Author: lxz-HXY 
#email: yingxh1995@aliyun.com

'''
MTCNN进行人脸检测、Deep-sort算法进行人脸跟踪
'''
from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np   
import gc
from multiprocessing import Process, Manager 

# 导入人脸检测需要的模块
from training.mtcnn_model import P_Net, R_Net,O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
# 导入跟踪算法的模块
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

'''
加载MTCNN检测模型
'''
def mtcnn(stage):
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = './tmp/model/pnet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net,modelPath) 
    if stage in ['rnet', 'onet']:
        modelPath = './tmp/model/rnet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = './tmp/model/onet/'
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath)
    return detectors

'''
python多进程：
receive：接收图片
realse：处理图片，进行人脸检测+跟踪
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
                del stack[:]
                gc.collect()


def realse(stack):
    # 解决GPU内存占用问题
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    # deep_sort 
    model_filename = 'mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    # mtcnn
    detectors = mtcnn('onet')
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])
    while True:
        if len(stack) > 0:
            frame = stack.pop()
            frame = cv2.resize(frame, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
            frame = np.array(frame)

            # 输出tmp信息为[x,y,w,h]; x:检测框左上角点的x坐标；y:检测框左上角y坐标；w:框宽;h:框高；
            # 原MTCNN输出的信息为检测框一对角点的坐标信息（左上角点、右下角点），以及检测为人脸的概率值[x1,y2,x2,y2,置信度]；
            tmp, _ = mtcnnDetector.detect_follow(frame)

            features = encoder(frame, tmp)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(tmp, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 1)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])), 0, 0.5, (0,255,0), 2)
            print('the id is: {}'.format(track.track_id))
            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 1)

            cv2.imshow("Detected", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config = config)
    
    t = Manager().list()
    t1 = Process(target=receive, args=(t,))
    t2 = Process(target=realse, args=(t,))
    t1.start()
    t2.start()
    t1.join()
    t2.terminate()
