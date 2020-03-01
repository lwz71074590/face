import sys
import os
import unittest
import time
import cv2

here = os.path.abspath(os.path.dirname(__file__))

# 将所要测试的源码路径放入path下面
source_path = os.path.join(here, '../')
sys.path.append(source_path)

from algorithm import insightface
from utils.decorator import excution_time


class TestInsightFace(unittest.TestCase):

    def setUp(self):
        self.test_image = cv2.imread(os.path.join(
            source_path, 'database/cache/test_picture.png'))
        self.database_path = os.path.join(source_path, "database/origin")
        self.muti_face_database_path = os.path.join(
            source_path, "database/old")

    @excution_time(10)
    def test_cosine_engine(self):

        cos_engine = insightface.CosineSimilarityEngine(0)
        cos_engine.load_database(self.database_path)

        for _ in range(10):
            _, names, p, _, _ = cos_engine.detect_recognize(self.test_image)
            print(names)

    @excution_time(10)
    def test_nearest_neibor(self):
        nearest_engine = insightface.NearestNeighborsEngine(0)
        nearest_engine.load_database(self.database_path)

        for _ in range(10):
            _, names, p, _, _ = nearest_engine.detect_recognize(
                self.test_image)
            pass

    def test_svm_engine(self):
        svm_engine = insightface.SVMClassificationEngine(0)
        svm_engine.load_database(self.muti_face_database_path)
        
        for _ in range(10):
            _, names, p, _, _ = svm_engine.detect_recognize(
                self.test_image)
            print(names)
            pass
