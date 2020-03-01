"""
人脸识别接口
"""

import os
import pickle
import glob
import numpy as np
import cv2
import mxnet as mx
import sys
import scipy
import sklearn

from sklearn import svm
from sklearn.model_selection import train_test_split
from gluonnlp.embedding.evaluation import CosineSimilarity
from sklearn.neighbors import NearestNeighbors

here = os.path.abspath(os.path.dirname(__file__))

sys.path.append(here)
import deploy.face_model as face_model
from .utils import load_dataset


class ModelParam(object):
    """初始化模型的一些参数
    """

    image_size = "112,112"
    model = os.path.join(
        here, 'models/model-r100-ii/model, 0')  # 预训练模型的保存路径
    ga_model = ''  # 性别年龄模型，在这里没有用到，为空
    gpu = 0  # gpu的id
    det = 0  # mtcnn option, 1 means using R+O, 0 means detect from begining
    flip = 0  # whether do lr flip aug
    threshold = 1.24  # ver dist threshold


class BaseEngine(object):
    """人脸识别引擎基类，提供模型初始化、提取特征向量的方法

    """

    def __init__(self, gpu_id=0):

        # 初始化模型的一些参数
        ModelParam.gpu = gpu_id
        self.model = face_model.FaceModel(ModelParam)
        self.database = None
        self.index2name = []
        self.feature_matrix = None

    def load_database(self, path, force_reload=False, save_intermediate_result=True, suffix='jpg'):
        """加载人脸库

        Args:
            path (str): 人脸库文件夹的绝对路径。路径下每个人的照片保存到以名字命名的文件夹中, 图片以jpg的格式存储
            force_reload (bool, optional): Defaults to False. 是否强制重新加载
            save_intermediate_result (bool, optional): Defaults to False. 是否保存生成的向量到embedding_matrix.npy。保存生成的向量可以提高下次加载模型的速度
        """
        self.feature_matrix, self.index2name, _ = load_dataset(
            path, self.model, force_reload=force_reload, save_intermediate_result=save_intermediate_result, suffix=suffix)

        # 将特征矩阵库加载到MXnet
        self.feature_matrix = mx.nd.array(
            self.feature_matrix, ctx=mx.gpu(ModelParam.gpu))

    def get_detection(self, img):
        """人脸检测。从照片中获取检测到的人脸

        Args:
            img (np.array|str): 图片矩阵。可以是从视频中获取的帧|图片路径
        """
        # 如果传入的是图片路径，则从图片路径中加载图片
        if isinstance(img, str):
            img = cv2.imread(img)
        scaled_images, boxes, points, flag = self.model.get_input(img)

        return scaled_images, boxes, points, flag

    def recognize(self, scaled_images):
        """给定人脸图片集合，返回识别的姓名，概率和向量.
            需要在子类实现

        Args:
            scaled_images (list)): 人脸图片矩阵列表
        """

        raise NotImplementedError

    def detect_recognize(self, img, batch_size=5, p_threshold=-1, min_size=0):
        """人脸检测+识别

        Args:
            img (np.array|str): 图片矩阵。可以是从视频中获取的帧|图片路径
            batch_size (int): batch caculation in order to avoid "cuda out of memory" error.
            p_threshold (float): min probability for recognize.
            min_size (int): The detected face smaller than <min_size> will be filtered out.
        """
        scaled_images, boxes, points, flag = self.get_detection(img)

        # 什么都没检测到，返回空
        if not flag:
            return [], [], [], [], []

        original_face_image = list()
        boxes = boxes.astype(np.int32)

        # filter images that don't meet the requirement out.
        size_filter = []
        for i, b in enumerate(boxes):
            if b[3] - b[1] > min_size and b[2] - b[0] > min_size:
                size_filter.append(i)

        scaled_images = scaled_images[size_filter]
        boxes = boxes[size_filter]

        # 如果数据量过大，分batch防止显存溢出
        loop_num = (scaled_images.shape[0]-1)//batch_size + 1
        names, probabilities = [], []
        for i in range(loop_num):
            if (i+1) * batch_size > scaled_images.shape[0]:
                input_matrix = scaled_images[i*batch_size:]
            else:
                input_matrix = scaled_images[i*batch_size: (i+1)*batch_size]

            n, p = self.recognize(input_matrix)
            names.extend(n)
            probabilities.extend(p)

        # probablities filter
        probabilities = np.array(probabilities)
        if p_threshold > -1:
            p_filter = probabilities > p_threshold
            names = [n for n, flag in zip(names, p_filter) if flag]
            if len(names) == 0:
                return [], [], [], [], []
            probabilities = probabilities[p_filter]
            boxes = boxes[p_filter]
            points = np.concatenate([p for p, flag in zip(points, p_filter) if flag])
            

        for box in boxes:
            original_face_image.append(img[box[1]: box[3], box[0]: box[2]])

        return original_face_image, names, probabilities, boxes, points

    def detect_recognize_stranger(self, img, p_threshold, p_threshold_stranger, min_size, batch_size=5, stranger_id='stranger'):
        """
        人脸识别+陌生人识别，必须指定阈值
        """
        original_face_image, names, probabilities, boxes, _ = self.detect_recognize(
            img, min_size=min_size, batch_size=batch_size)

        if len(names) == 0:
            acquaintance = {
                'image_matrix': [],
                'names': [],
                'probabilities': [],
                'boxes': []
            }
            stranger = {
                'image_matrix': [],
                'names': [],
                'probabilities': [],
                'boxes': []
            }
            return acquaintance, stranger

        # filter out acquaintance
        p_filter = probabilities > p_threshold
        n_acquaintance = [n for n, flag in zip(names, p_filter) if flag]
        if len(n_acquaintance) == 0:
            acquaintance = {
                'image_matrix': [],
                'names': [],
                'probabilities': [],
                'boxes': []
            }
        else:
            p_acquaintance = probabilities[p_filter]
            b_acquaintance = boxes[p_filter]
            image_acquaintance = [im for im, flag in zip(
                original_face_image, p_filter) if flag]

            acquaintance = {
                'image_matrix': image_acquaintance,
                'names': n_acquaintance,
                'probabilities': p_acquaintance,
                'boxes': b_acquaintance
            }

        # filter out stranger
        p_filter = probabilities < p_threshold_stranger
        n_stranger = [stranger_id for n, flag in zip(names, p_filter) if flag]

        if len(n_stranger) == 0:
            stranger = {
                'image_matrix': [],
                'names': [],
                'probabilities': [],
                'boxes': []
            }
        else:
            p_stranger = probabilities[p_filter]
            b_stranger = boxes[p_filter]
            image_stranger = [im for im, flag in zip(
                original_face_image, p_filter) if flag]

            stranger = {
                'image_matrix': image_stranger,
                'names': n_stranger,
                'probabilities': p_stranger,
                'boxes': b_stranger
            }

        return acquaintance, stranger

    def visualize(self, image, names, probabilities, boxes, points=None):
        for name, p, box in zip(names, probabilities, boxes):
            cv2.rectangle(image, (box[0], box[1]),
                          (box[2], box[3]), (255, 0, 0), 2)

            cv2.putText(image, '%s: %f' % (
                name, p), (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if points is not None:
            for p in points:
                for i in range(5):
                    cv2.circle(image, (p[i], p[i+5]), 1, (0, 255, 0), -1)

        return image


class CosineSimilarityEngine(BaseEngine):
    """通过余弦相似度进行人脸识别的引擎
    """

    def __init__(self, *args, **kwargs):
        super(CosineSimilarityEngine, self).__init__(*args, **kwargs)
        self.cos_op = CosineSimilarity()

    def recognize(self, scaled_images):
        """Hook implementation of super class
        """
        mx_image_tensor = self.model.get_feature_tensor(scaled_images)
        result = self._comput_cos_similarity(
            mx_image_tensor, self.feature_matrix)
        return result

    def _comput_cos_similarity(self, a, b):
        """计算两个矩阵每个向量间的余弦相似度，可能会显存溢出，推荐batch大小为10个

        Args:
            a (mx.NDArray):
            b (mx.NDArray):
        """

        a_ = a.expand_dims(1).broadcast_axes(
            1, b.shape[0]).reshape(-1, a.shape[-1])
        b_ = b.expand_dims(0).broadcast_axes(
            0, a.shape[0]).reshape(-1, b.shape[-1])
        batch_similarities = self.cos_op(
            a_, b_).reshape(a.shape[0], b.shape[0])
        best_similarities = batch_similarities.max(1).asnumpy()
        best_index = batch_similarities.argmax(1).asnumpy().astype(np.int32)
        names = [self.index2name[i] for i in best_index]

        return names, best_similarities


class CosineVoteEngine(BaseEngine):
    """通过余弦相似度加投票的方法进行人脸识别
    """

    def __init__(self, top=5, **kwargs):
        super(CosineVoteEngine, self).__init__(**kwargs)
        self.cos_op = CosineSimilarity()
        self.top = top

    def recognize(self, scaled_images):
        """Hook implementation of super class
        """
        mx_image_tensor = self.model.get_feature_tensor(scaled_images)
        names, probabilities = self._comput_cos_similarity(
            mx_image_tensor, self.feature_matrix)
        return names, probabilities

    def _comput_cos_similarity(self, a, b):
        """计算两个矩阵每个向量间的余弦相似度, 可能会显存溢出，推荐batch大小为10个

        Args:
            a (mx.NDArray):
            b (mx.NDArray):
        """

        a_ = a.expand_dims(1).broadcast_axes(
            1, b.shape[0]).reshape(-1, a.shape[-1])
        b_ = b.expand_dims(0).broadcast_axes(
            0, a.shape[0]).reshape(-1, b.shape[-1])
        batch_similarities = self.cos_op(
            a_, b_).reshape(a.shape[0], b.shape[0])
        best_index = batch_similarities.topk(
            1, k=self.top).asnumpy().astype(np.int32)
        names = []
        probabilities = []

        for i, item in enumerate(best_index):
            tmp = dict()
            for index in item:
                name = self.index2name[index]
                if name in tmp:
                    tmp[name] += batch_similarities[i][index]
                else:
                    tmp[name] = batch_similarities[i][index]
            sorted_name = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
            names.append(sorted_name[0][0])
            probabilities.append(sorted_name[0][1])
        probabilities = np.array(probabilities)

        return names, probabilities


class NearestNeighborsEngine(BaseEngine):

    """使用sklean里面NearestNeighbors（基于kdtree或balltree）来计算与库中最相似的人脸 TODO
    """

    def load_database(self, *args, **kwargs):
        super(NearestNeighborsEngine, self).load_database(*args, **kwargs)
        self.feature_matrix = sklearn.preprocessing.normalize(
            self.feature_matrix.asnumpy())
        self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.feature_matrix)

    def recognize(self, scaled_images):
        """Hook implementation of super class
        """
        image_tensor = self.model.get_feature(scaled_images)
        dist, indices = self.nbrs.kneighbors(image_tensor)
        indices = np.squeeze(indices)
        dist = np.squeeze(dist)

        names = [self.index2name[i] for i in indices]
        return names, dist


class SVMClassificationEngine(BaseEngine):
    """使用sklean里面SVM分类器来计算与库中最相似的人脸
    """

    def __init__(self, gpu_id=0, pre_trained=os.path.join(here, 'pre_trained/svm_cls/model.pkl'), force_reload=False):
        super(SVMClassificationEngine, self).__init__(gpu_id=gpu_id)
        self.pre_trained = pre_trained
        self.force_reload = force_reload

        # Create it if directory are not exists.
        if not os.path.isdir(os.path.dirname(pre_trained)):
            os.mkdir(os.path.dirname(pre_trained))

    def load_database(self, *args, **kwargs):
        super(SVMClassificationEngine, self).load_database(*args, **kwargs)
        self.feature_matrix = sklearn.preprocessing.normalize(
            self.feature_matrix.asnumpy())

        label2name = list()
        for n in self.index2name:
            if n not in label2name:
                label2name.append(n)

        self.label2name = label2name
        self.name2label = {name: i for i, name in enumerate(label2name)}

        self.labels = np.array([self.name2label[n] for n in self.index2name])

        if not os.path.exists(self.pre_trained) or self.force_reload:
            self.clf = svm.SVC(
                kernel='linear', probability=True, decision_function_shape="ovr")
            self.clf.fit(self.feature_matrix, self.labels)
            f = open(self.pre_trained, 'wb')
            pickle.dump(self.clf, f)
            f.close()
        else:
            f = open(self.pre_trained, 'rb')
            self.clf = pickle.load(f)
            f.close()

    def draw_box_diagram(self):
        # log_prob = self.clf.predict_log_proba(self.feature_matrix[:100])
        # log_prob = np.max(log_prob, 1)

        prob = self.clf.predict_proba(self.feature_matrix[:10])
        prob = np.max(prob, 1)
        y_hat = self.clf.predict(self.feature_matrix[:10])
        equal = y_hat == self.labels[:10]
        # log_prob = log_prob[equal]
        prob = prob[equal]

        means = np.mean(prob)
        std = np.std(prob)

        print("均值: %f." % means)
        print("方差: %f." % std)

        import pandas as pd
        
        import matplotlib.pyplot as plt
        data = pd.DataFrame({
            # "log_prob": log_prob,
            "prob": prob
        })

        data.boxplot()
        plt.ylabel("Probablity")
        plt.xlabel("Type")
        plt.show()

    def train_test(self, *args, **kwargs):
        super(SVMClassificationEngine, self).load_database(*args, **kwargs)
        self.feature_matrix = sklearn.preprocessing.normalize(
            self.feature_matrix.asnumpy())
 
        label2name = list()
        for n in self.index2name:
            if n not in label2name:
                label2name.append(n)

        self.label2name = label2name
        self.name2label = {name: i for i, name in enumerate(label2name)}

        self.labels = np.array([self.name2label[n] for n in self.index2name])

        x_train, x_test, y_train, y_test = train_test_split(
            self.feature_matrix, self.labels, random_state=25, train_size=0.8, test_size=0.2)

        self.clf = svm.SVC(
            kernel='linear', probability=True, decision_function_shape="ovr")
        self.clf.fit(x_train, y_train)

        f = open(self.pre_trained, 'wb')
        pickle.dump(self.clf, f)
        f.close()

        score = self.clf.score(x_test, y_test)
        return score

    def recognize(self, scaled_images):
        """Hook implementation of super class
        """
        image_tensor = self.model.get_feature(scaled_images)
        predictions = self.clf.predict_proba(image_tensor)

        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(
            len(best_class_indices)), best_class_indices]

        names = [self.label2name[i] for i in best_class_indices]
        return names, best_class_probabilities
