'''
@Author: TangZhiFeng
@Data: 2019-01-06
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 15:28:54
@Description: 专注度识别
'''
import numpy as np
import time


class Forcus(object):
    '''这是一个进程的专注度
    '''

    def __init__(self, camera2box):
        '''初始化

        Arguments:
            camera2box {dict} -- 多进程需要共享的dict
            look {进程锁} -- 多进程的LOOK
        '''

        self.camera2box = camera2box

    def get_forcus(self, camera_id, names, points):
        '''获取专注度

        Arguments:
            camera_id {str} -- 设备id
            names {list} -- 识别人员id的list
            box {numpy.array} -- 人员的
        '''
        # print('历史')
        # print(self.camera2box)
        # print('box')
        # print(points)
        _camera2box = self.camera2box.copy()
        if isinstance(points, list):
            return []
        if not camera_id in self.camera2box:
            last = {}
        else:
            last = _camera2box[camera_id]
        box = points.reshape(-1, 5, 2)
        current = {names[i]: {'box': box[i], 'update_time': time.time()}
                   for i in range(len(names))}
        _result = self.__calculation(current, last, names)
        # 这里是把数据进行更新
        updata = last
        for i in range(len(names)):
            name = names[i]
            # 这里会记录每一次的更新时间
            if name in updata:
                updata[name]['update_time'] = time.time()
                updata[name]['box'] = box[i]
            else:
                updata[name] = {}
                updata[name]['update_time'] = time.time()
                updata[name]['box'] = box[i]
        # 更新的时候加上锁，确保进程安全
        self.camera2box[camera_id] = updata
        return _result

    def __point_distance(self, last_point, current_point):
        '''两点的距离

        Arguments:
            last_point {numpy} -- 上一次的所有点
            current_point {numpy} -- 当前的点

        Returns:
            int -- 两点的欧氏距离
        '''

        result = np.linalg.norm(last_point-current_point)
        return result

    def __calculation(self, current, last, names):
        '''计算专注度

        Arguments:
            current {dict} -- 当前的对应数据
            last {dict} -- 历史对应的数据
            names {list} -- 人员list信息

        Returns:
            list -- 人员list的对应信息
        '''

        _forcus = []
        for name in names:
            if name in last:
                # 这里判断上一次的数据多少秒失效
                if time.time() - last[name]['update_time'] < 20:
                    box_current = current[name]['box']
                    box_last = last[name]['box']
                    # 获取左眼到右眼加上左眼到左嘴的距离
                    current_mouth_eye_dis = self.__point_distance(
                        box_current[0], box_current[1]) + self.__point_distance(box_current[0], box_current[3])
                    last_mouth_eye_dis = self.__point_distance(
                        box_last[0], box_last[1]) + self.__point_distance(box_last[0], box_last[3])
                    # print('当前距离：')
                    # print(current_mouth_eye_dis)
                    # print('上一次的距离：')
                    # print(last_mouth_eye_dis)
                    # 获取比例
                    zoom = last_mouth_eye_dis*1.0 / current_mouth_eye_dis
                    # print('缩放比例：')
                    # print(zoom)
                    # 进行缩放并返回距离
                    point_point_dis = self.__point_distance(
                        box_current * zoom, box_last)
                    # move_proportion = point_point_dis*1.0 / last_mouth_eye_dis
                    # print(point_point_dis)
                    # print(move_proportion)
                    if point_point_dis <= 80:
                        _result = 3
                    elif point_point_dis <=200:
                        _result = 2
                    else:
                        _result = 1
                else:
                    _result = 3
            else:
                _result = 3
            _forcus.append(_result)
        # TODO 这里要把_forcus进行处理
        return _forcus
