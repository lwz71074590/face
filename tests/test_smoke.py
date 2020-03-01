import os
import sys
import cv2
import unittest

here = os.path.abspath(os.path.dirname(__file__))

# 将所要测试的源码路径放入path下面
source_path = os.path.join(here, '../')
sys.path.append(source_path)
from algorithm.fire_detection.smoke_detect import detect, get_default_model


class TestFireDetection(unittest.TestCase):

    def test_smoke_detection(self):

        sess, model = get_default_model()

        cap = cv2.VideoCapture(os.path.join(
            source_path, 'database/cache/fire.mp4'))

        while True:
            res, frame = cap.read()

            if not res:
                continue
            # frame = os.path.join(here, '../database/cache/smoke.png')
            # frame = cv2.imread(frame)

            flag, frame, cameraKey, image_id = detect(
                sess, model, frame, frame.shape[0], frame.shape[1], 10, '1')

            assert not flag
            break
