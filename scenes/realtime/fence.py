'''
@Author: TangZhiFeng
@Data: 2019-01-05
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 14:21:27
@Description: 实时监控围墙
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
from processes.nodes.recorder import CameraReader
from scenes import BaseEngineering, BaseScenesManage
from processes.nodes.recognizer import RealTimeStrangerRecognizer

from utils.time_utils import genera_stamp
from utils.socket_client import client
from utils.upload_image import batch_people_upload, batch_stranger_upload

class RealTimeFenceEngineering(BaseEngineering):
    '''实时检测围墙
    '''

    def __init__(self):
        real_time = True
        super(RealTimeFenceEngineering,self).__init__(real_time)

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
            'scene': 105
        }}
        # TODO data.stranger_labels直接变成str类型
        stranger = data.stranger_labels
        stranger = [i.decode('utf-8') for i in stranger]
        result['data']['data']['camera_id'] = data.chanel_id
        result['data']['data']['scenario_id'] = data.chanel_id
        result['data']['data']['recognition'] = data.names
        result['data']['data']['stranger'] = stranger  
        paths = array_to_file(data.image_matrix, data.names)
        batch_people_upload(paths, data.chanel_id,
                            data.names, result['data']['stamp'])
        paths = array_to_file(data.stranger_matrix, data.stranger_labels)
        batch_stranger_upload(paths,stranger)
        client.send(json.dumps(result, ensure_ascii=False))


class RealTimeFenceScene(BaseScenesManage):
    '''实时检查围墙
    '''

    def init_node(self):
        camera = CameraReader()
        real_time_rec = RealTimeStrangerRecognizer(1)
        diff = FrameDiffNode(1)
        diff.init_node()
        camera_ip = ['rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101'] 
        camera_id = [camera_ip2camera_id[i] for i in camera_ip]
        camera.init_node(camera_ip, camera_id, 40, "123")
        real_time_rec.init_node()
        self.nodes.append(camera)
        self.nodes.append(diff)
        self.nodes.append(real_time_rec)
        self.manage = RealTimeFenceEngineering()


if __name__ == "__main__":
    rt_scene = RealTimeFenceScene()
    rt_scene.start()
