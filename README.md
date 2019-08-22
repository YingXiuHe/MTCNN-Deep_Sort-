# MTCNN-Deep_Sort
MTCNN + Deep_Sort算法进行人脸跟踪

# MtCNN算法
# Dependencies
* Tensorflow v1.0.0 or higher
* TF-Slim
* Python 2.7/3.5
* Ubuntu 14.04/16.04
* cuda 10+cudnn7.5+tf-1.13.1

# 相关代码：
 1.mtcnn人脸检测模型保存在tmp/model目录下
 2.相关代码
  视频人脸检测代码：testing/camera.py 

# Deep_Sort算法
# Dependencies
    python2.7/3
    NumPy
    sklean
    OpenCV
    Pillow
  Additionally, feature generation requires TensorFlow.

# 相关代码
 1.结合mtcnn人脸检测跟踪的代码： mtcnn_sort.py
 2.跟踪算法的模型：mars-small128.pb


# 以上涉及的脚本文件，环境配置好后，直接运行即可；
