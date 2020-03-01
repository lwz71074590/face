'''
@Author: TangZhiFeng
@Data: 2019-01-04
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 00:44:50
@Description: 火焰检测的接口 
'''
import os
import sys
current = os.path.dirname(__file__)
project = os.path.dirname(os.path.dirname(current))
sys.path.append(project)
import cv2
import numpy as np
from .firenet import get_model




class FireParam(object):
    '''初始火焰检测的参数
    '''

    network_rows = 224
    network_cols = 224


class FireEngine(object):

    def __init__(self):
        # self.model = construct_firenet(FireParam.network_rows, FireParam.network_cols)
        # self.model.load(os.path.join(current, "models/FireNet",
        #                              "firenet"), weights_only=True)
        self.model = get_model()

    def load_model(self):
        '''加载模型
        '''

        self.model.load(os.path.join(current, "models/FireNet",
                                     "firenet"), weights_only=True)

    def predict(self, frame):
        '''预测图像中是否有火情

        Arguments:
            frame {ndarray} -- 一张图片为ndarray格式

        Returns:
            bool -- True为有火情，False为没有火情
        '''

        small_frame = cv2.resize(
            frame, (FireParam.network_rows, FireParam.network_cols), cv2.INTER_AREA)
        output = self.model.predict([small_frame])

        return round(output[0][0]) == 1
