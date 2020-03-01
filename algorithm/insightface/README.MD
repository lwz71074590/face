# 基于MTCNN的人脸检测算法，基于ArcFace的人脸识别算法
本模块基于开源的mxnet实现： [insightface](https://github.com/deepinsight/insightface)

## 使用说明
根据计算人脸向量相似度算法的不同，该模块目前实现了三个Engine。


##### CosineSimilarityEngine
使用暴力计算（brute force）。算法复杂度o(n*m)
```py
import cv2
from insightface import CosineSimilarityEngine



engine = CosineSimilarityEngine(gpu_id=0)
engine.load_database('../database/origin', True)

frame = cv2.imread('your_img.jpg')
original_face_image, names, probabilities, boxes, points = engine.detect_recognize(frame, p_threshold=self.threshold, min_size=self.minsize)
```

##### CosineVoteEngine
使用投票的方式计算最近的一个人脸。（数据集中每个人有多张人脸数据可以使用）算法复杂度o(n*m)

```py
import cv2
from insightface import CosineSimilarityEngine

engine = CosineSimilarityEngine(top=5, gpu_id=0)
engine.load_database('../database/origin', True)

frame = cv2.imread('your_img.jpg')
original_face_image, names, probabilities, boxes, points = engine.detect_recognize(frame, p_threshold=self.threshold, min_size=self.minsize)
```

##### NearestNeighborsEngine
使用sklearn 近邻算法计算最相似的人脸。（内部使用ball-tree，kdtree）
```py
import cv2
from insightface import NearestNeighborsEngine

engine = CosineSimilarityEngine(gpu_id=0)
engine.load_database('../database/origin', True)

frame = cv2.imread('your_img.jpg')
original_face_image, names, probabilities, boxes, points = engine.detect_recognize(frame, p_threshold=self.threshold, min_size=self.minsize)
```

##### SVMClassificationEngine (待完成，暂时不可用)

## 其他功能
##### 可视化识别结果
将检测到的人脸和识别结果显示在图片上
```py
processed_image = engine.visualize(self, image, names, probabilities, boxes, points)  # image是原始图片， names、probabilities、boxes 是detect_recognize返回的结果
```
