'''
@Author: TangZhiFeng
@Data: 2019-01-08
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 15:11:50
@Description: 教室实时识别
'''

import os
import sys
import json

current = os.path.dirname(__name__)
project = os.path.dirname(os.path.dirname(current))
sys.path.append(project)
from utils.keymap import camera_ip2camera_id
from utils.image_base64 import array_to_file
from processes.nodes.diff_node import FrameDiffNode
from processes.nodes.recognizer import AttentionRecognizer
from processes.nodes.recorder import CameraReader
from scenes import BaseEngineering, BaseScenesManage
from processes.nodes.recognizer import RealTimeRecognizer,RealTimeStrangerRecognizer
from utils.time_utils import genera_stamp
from utils.socket_client import client
from utils.upload_image import batch_people_upload

class RealTimeClassroomEngineering(BaseEngineering):
    '''实时检测教室出入大门
    '''

    def __init__(self):
        real_time = True
        super(RealTimeClassroomEngineering, self).__init__(real_time)

    def build_data(self, i, data):
        return data

    def generater(self, data):
        '''返回示例{
                data:{
                    type:1//识别数据类型,1 为人员检测结果,非 1 为场景异常结果
                    data:{
                        camera_id:xxxx,//识别摄像头 id
                        scenario_id:xxx,//场所 id
                        recognition:[xxxxx,xxxxx,xxxxx],//图片识别时间戳
                        stranger:[xxxxx,xxxxx,xxxxx] //陌生人
                    },
                    stamp:xxxxx,
                    //识别结果时间戳
                    scene:xxxx
                    //识别场景类型,不同场景类型对应不同异常规则,scene>=100
                }
            }

        Arguments:
            data {[obj]} -- 对象中含有我需要的数据
        '''

        result = {'data': {
            'type': 1,
            'data': {},
            'stamp': genera_stamp(),
            'scene': 103
        }}
        result['data']['data']['camera_id'] = data.chanel_id
        result['data']['data']['scenario_id'] = data.chanel_id
        result['data']['data']['recognition'] = data.names
        result['data']['data']['attention'] = data.score
        # path = array_to_file(data.image_matrix, data.names)
        # batch_people_upload(path, data.chanel_id,
        #                     data.names, result['data']['stamp'])
        client.send(json.dumps(result, ensure_ascii=False))


class RealTimeClassroomScene(BaseScenesManage):
    '''实时教室检测
    '''

    def init_node(self):
        camera_ip = ['rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101',
                     'rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101']
        camera_id = [camera_ip2camera_id[i] for i in camera_ip]
        camera = CameraReader()
        attention = AttentionRecognizer(1)
        diff = FrameDiffNode(1)
        diff.init_node()
        attention.init_node()
        camera.init_node(camera_ip, camera_id, 20, "123")

        camera.run()
        diff.run()

        attention.run()
        self.nodes.append(camera)
        self.nodes.append(diff)

        self.nodes.append(attention)
        self.manage = RealTimeClassroomEngineering()


if __name__ == "__main__":
    rt_scene = RealTimeClassroomScene()
    rt_scene.start()

