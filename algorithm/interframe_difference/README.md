# 帧间差分算法 （王宁）

&ensp;&ensp;&ensp;&ensp;帧间差分法是一种通过对视频图像序列中相邻两帧作差分运算来获得运动目标轮廓的方法，它可以很好地适用于存在多个运动目标和摄像机移动的情况。当监控场景中出现异常物体运动时，帧与帧之间会出现较为明显的差别，两帧相减，得到两帧图像亮度差的绝对值，判断它是否大于阈值来分析视频或图像序列的运动特性，确定图像序列中有无物体运动。只在有物体时进行人脸识别或者物体检测

# 用法
```python
import cv2
from interframe_difference import FrameDiff

frame = cv2.imread('Your_Image_File.jpg')  # 也可以从视频中读取
discriminator = FrameDiff()

# 返回True表示有运动物体或者像素变化超过阈值，否则返回False
ret = discriminator.compute_diff(frame)
```