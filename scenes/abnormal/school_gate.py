'''
@Author: TangZhiFeng
@Data: 2019-01-04 
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 16:35:42
@Description: 大门异常检测
'''

import os
import sys
import json

current = os.path.dirname(__name__)
project = os.path.dirname(os.path.dirname(current))
sys.path.append(project)

from utils.upload_image import batch_people_upload, batch_type2_upload
from utils.keymap import camera_ip2camera_id, abnormal_type_to_rule_id
from processes.nodes.recognizer import AbnormalDetectionRecognizer
from utils.socket_client import client
from utils.time_utils import genera_stamp
from processes.nodes.recognizer import RealTimeRecognizer
from scenes import BaseEngineering, BaseScenesManage
from processes.nodes.recorder import CameraReader
from processes.nodes.fire_node import FlameDiffNode
from processes.nodes.diff_node import FrameDiffNode
from utils.image_base64 import array_to_file
class AbnormalSchoolGateEngineering(BaseEngineering):
    '''实时检测学校出入大门异常情况
    '''

    def __init__(self):
        real_time = True
        super(AbnormalSchoolGateEngineering, self).__init__(real_time)

    def build_data(self, i, data):
        return data

    def generater(self, data):
        '''返回示例{
            type:x
                //识别数据类型,1 为人员检测结果,非 1 为场景异常结果
                data:{
                    camera_id:xxxx,
                    //识别摄像头 id
                    scenario_id:xxx,
                    //场所 id
                    image_id:xxxxx ,
                    //图片识别时间戳
                    rule_id:xxxx
                    //匹配的异常规则
                }
            }

        Arguments:
            data {[obj]} -- 对象中含有我需要的数据,stay_too_long逗留，cluster聚集
        '''
        result = {'data': {
            'type': 2,
            'data': {}
        }}
        stamp = genera_stamp()
        rule_id = abnormal_type_to_rule_id[data.abnormal_type]
        result['data']['data']['camera_id'] = data.camera_key
        result['data']['data']['scenario_id'] = data.camera_key
        result['data']['stamp'] = stamp
        result['data']['rule_id'] = rule_id
        # 这里后面一个参数只是取这个list的长度作为生成paths的list长度，前者需要一个list。
        path = array_to_file([data.base64_data], [1])
        assert len(path) == 1
        batch_type2_upload(path[0], data.camera_key, result['data']['stamp'])
        client.send(json.dumps(result, ensure_ascii=False))


class AbnormalSchoolGateScene(BaseScenesManage):
    '''实时校门异常检测
    '''

    def init_node(self):
        camera = CameraReader()
        realtime_abnormal = AbnormalDetectionRecognizer()
        diff = FrameDiffNode(1)
        diff.init_node()
        flame = FlameDiffNode()
        camera_ip = [
            'rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101','rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101']
        camera_id = [camera_ip2camera_id[ip] for ip in camera_ip]
        # camera_id = ['1']
        camera.init_node(camera_ip, camera_id, 100, "123")
        realtime_abnormal.init_node()
        flame.init_node()
        camera.run()
        diff.run()
        realtime_abnormal.run()
        flame.run()
        self.nodes.append(camera)
        self.nodes.append(diff)
        self.nodes.append([realtime_abnormal, flame])
        self.manage = AbnormalSchoolGateEngineering()


if __name__ == "__main__":
    rt_scene = AbnormalSchoolGateScene()
    rt_scene.start()
